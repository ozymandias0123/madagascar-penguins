"""
agents/news_analyst.py — Rico (News / Macro Analyst)
Model  : Gemini 1.5 Flash  (Google AI)
Role   : The wild card. Invoked ONLY when Kowalski sets needs_news_check=True.
         Fetches REAL headlines + economic calendar from free sources,
         then passes everything to Gemini for analysis.
Node   : rico_news_node  →  updates state['news_analysis']

Free news sources used:
  - RSS feeds (Reuters, CNBC, MarketWatch, Yahoo Finance, Investing.com)
  - ForexFactory economic calendar scrape
  - Finnhub free tier (if FINNHUB_API_KEY set in .env)
  - Yahoo Finance via yfinance
"""

import logging
from datetime import datetime
from typing import Dict

from config import Config
from agents.base_agent import BaseAgent

# News fetcher — graceful if unavailable
try:
    from utils.news_fetcher import get_fetcher
    _NEWS_AVAILABLE = True
except Exception:
    _NEWS_AVAILABLE = False


SYSTEM_PROMPT = """\
You are Rico — the wild, unpredictable member of the Penguin Squad.
Your job: assess macro and session risk for NAS100 / USTECm trades.
You eat danger for breakfast, but even YOU know when the market is a trap.

Given session timing, day of week, and price-action context,
decide whether it is SAFE to trade right now.

HIGH-RISK conditions (set safe_to_trade=false):
  - Friday after 15:00 UTC          (NFP risk / weekend gap)
  - Wednesday 18:00-19:00 UTC       (FOMC release window)
  - Monday 00:00-07:00 UTC          (weekend gap open)
  - ATR ratio > 2.5                 (extreme volatility spike)
  - ADX < 15                        (dead ranging market)
  - Outside all kill zones with no Silver Bullet confirmation

SESSION QUALITY GUIDE:
  premium  -- Silver Bullet window (15-16 UTC / 19-20 UTC) or London 07-09 UTC
  standard -- NY Kill Zone 13-16 UTC
  low      -- all other hours

RESPOND ONLY with this exact JSON (no markdown, no extra text):
{
  "session_sentiment":       "risk_on|risk_off|neutral",
  "macro_risk_level":        "low|medium|high",
  "volatility_expectation":  "low|normal|high",
  "news_event_risk":         false,
  "safe_to_trade":           true,
  "session_quality":         "premium|standard|low",
  "confidence":              75,
  "notes":                   "one concise Rico-style sentence"
}"""


class Rico(BaseAgent):
    """
    Rico — News / Macro Analyst powered by Gemini 1.5 Flash.
    The wild card who sniffs out dangerous market conditions.
    Invoked only when Kowalski requests a news check.
    """
    name = "Rico"

    def __init__(self):
        if not Config.GOOGLE_API_KEY:
            logging.warning("[Rico] GOOGLE_API_KEY not set — will use fallback")
        self._model = None

    def _get_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self._model = genai.GenerativeModel(Config.GEMINI_MODEL)
        return self._model

    # -- LLM call ------------------------------------------------

    def _call_llm(self, state: Dict) -> str:
        if not Config.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY missing")

        model  = self._get_model()
        ctx    = state.get('context') or {}
        ts_str = state.get('timestamp', datetime.utcnow().isoformat())
        try:
            ts   = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            dow  = ts.strftime('%A')
            hour = ts.hour
        except Exception:
            dow, hour = 'Unknown', state.get('hour', 12)

        ra           = state.get('risk_assessment') or {}
        kow_risk     = ra.get('risk_level', 'unknown')
        kow_warnings = ra.get('warnings', [])

        # ── Fetch real news ──────────────────────────────────────
        news_block = ""
        if _NEWS_AVAILABLE:
            try:
                fetcher    = get_fetcher()
                symbols    = Config.SYMBOLS or ["USTECm"]
                news_block = fetcher.get_summary(
                    symbols=symbols,
                    hours_ahead=12,
                )
                logging.info("[Rico] 📰 Live news fetched successfully")
            except Exception as exc:
                logging.warning(f"[Rico] News fetch failed: {exc}")
                news_block = "⚠️  News fetch failed — using timing analysis only."
        else:
            news_block = "⚠️  news_fetcher not available — reasoning from timing only."

        user_msg = f"""\
Rico — Kowalski needs your nose on this one. Sniff out the macro danger.
Kowalski's risk level : {kow_risk}
Kowalski's warnings   : {kow_warnings}

TIMING:
  Day           : {dow}
  UTC Hour      : {hour:02d}:00
  Session       : {state.get('session', 'new_york')}
  Silver Bullet : {state.get('is_silver_bullet', False)}
  Timestamp     : {ts_str}

MARKET CONDITIONS:
  ADX           : {ctx.get('adx', 20):.1f}  ({'TRENDING' if ctx.get('adx', 0) >= 25 else 'RANGING'})
  ATR Ratio     : {ctx.get('atr_ratio', 1.0):.2f}
  Volatility    : {ctx.get('volatility', 'normal').upper()}
  Regime        : {ctx.get('regime', 'trending').upper()}
  Zone Type     : {state.get('zone_type', 'equilibrium')}

SIGNAL:
  Direction     : {(state.get('signal') or {}).get('type', 'unknown').upper()}
  Quality Score : {state.get('quality_score', 0):.1f}/10
  HTF Bias      : {state.get('htf_bias', 'neutral')}
  Structure     : {state.get('structure', 'none')}

{news_block}

Rico — is the macro environment safe to trade right now?"""

        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\n{user_msg}",
            generation_config={"temperature": 0.1, "max_output_tokens": 400},
        )
        return response.text

    # -- State update --------------------------------------------

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        safe = result.get('safe_to_trade', True)
        risk = result.get('macro_risk_level', 'medium')
        logging.info(
            f"[Rico] 💣 "
            f"Safe={safe} | Risk={risk} | "
            f"Sentiment={result.get('session_sentiment')} | "
            f"Quality={result.get('session_quality')} | "
            f"\"{result.get('notes', '')}\""
        )
        return {'news_analysis': result}

    # -- Fallback ------------------------------------------------

    def _fallback(self, state: Dict) -> Dict:
        """Rico's gut feeling when Gemini API is down."""
        hour  = state.get('hour', 12)
        is_sb = state.get('is_silver_bullet', False)
        ctx   = state.get('context') or {}
        atr_r = ctx.get('atr_ratio', 1.0)

        safe  = True
        risk  = 'medium'
        notes = 'Gemini API down — Rico going on gut instinct'

        if atr_r > 2.5:
            safe  = False
            risk  = 'high'
            notes = 'Extreme volatility — even Rico says no'
        elif hour in range(0, 7):
            risk  = 'medium'
            notes = 'Early UTC hours — low liquidity, watch out'

        quality = 'premium' if is_sb else ('standard' if 13 <= hour <= 16 else 'low')

        return {
            'news_analysis': {
                'session_sentiment':      'neutral',
                'macro_risk_level':       risk,
                'volatility_expectation': 'high' if atr_r > 2.0 else 'normal',
                'news_event_risk':        False,
                'safe_to_trade':          safe,
                'session_quality':        quality,
                'confidence':             45,
                'notes':                  notes,
            }
        }
