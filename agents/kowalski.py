"""
agents/kowalski.py — Kowalski (Risk Manager)
Model  : Claude Sonnet
Role   : The tallest and smartest strategist. Calculates risk, position size,
         checks drawdown, reward:risk ratio, and issues a safety assessment.
         Decides whether a news check is needed before final validation.
Node   : kowalski_node  →  updates state['risk_assessment']
"""

import logging
from typing import Dict, List

from config import Config
from agents.base_agent import BaseAgent

try:
    from utils.news_fetcher import get_fetcher as _get_news
    _HAS_NEWS = True
except Exception:
    _HAS_NEWS = False


SYSTEM_PROMPT = """\
You are Kowalski — the tall, brilliant risk strategist of the trading team.
You calculate everything precisely. No guesswork. Pure risk mathematics.

YOUR JOB:
1. Verify Skipper's analysis makes sense from a risk perspective.
2. Calculate whether the RR ratio is acceptable for the account size.
3. Check all hard risk rules — reject immediately on any violation.
4. Decide if a macro/news check is necessary before validation.
5. Recommend position size adjustment if conditions are borderline.

HARD REJECTION RULES (any one → approved=false, no exceptions):
  R1. ADX < 25 AND NOT in Silver Bullet window
  R2. Confluence score < 6 (from Skipper)
  R3. Trade direction conflicts with H1 HTF bias
       (exception: very strong BOS with structure_quality="strong")
  R4. Hour outside kill zones: 13-16 UTC or 19-20 UTC Silver Bullet
  R5. Volatility=HIGH with no BOS confirmation (CHoCH alone is insufficient)
  R6. Account balance < $30
  R7. Skipper recommended HOLD

NEEDS_NEWS_CHECK = true when:
  • macro_risk_hour: hour in [14, 15, 19] (typically news-active)
  • risk_level = "high" (borderline approval)
  • ATR ratio > 1.8 (unusual volatility spike)
  • It is a Wednesday (FOMC) or Friday (NFP risk)

RISK LEVELS:
  low    — all rules pass, confluence ≥ 7, ADX ≥ 30
  medium — rules pass but some metrics are borderline
  high   — approved but with significant caveats (→ needs_news_check=true)

RESPOND ONLY with this exact JSON (no markdown, no extra text):
{
  "approved":           false,
  "rejection_reasons":  [],
  "risk_level":         "low|medium|high",
  "position_size_adj":  1.0,
  "max_risk_percent":   0.005,
  "needs_news_check":   false,
  "warnings":           [],
  "rr_valid":           true,
  "reasoning":          "one precise Kowalski-style sentence"
}"""


class Kowalski(BaseAgent):
    """
    Kowalski — Risk Manager powered by Claude Sonnet.
    The strategist who protects the account with strict mathematical rules.
    """
    name = "Kowalski"

    def __init__(self):
        if not Config.ANTHROPIC_API_KEY:
            logging.warning("[Kowalski] ANTHROPIC_API_KEY not set — will use fallback")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        return self._client

    # ── LLM call ─────────────────────────────────────────────

    def _call_llm(self, state: Dict) -> str:
        if not Config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY missing")

        ctx  = state.get('context') or {}
        ma   = state.get('market_analysis') or {}
        hour = state.get('hour', 0)

        # Day-of-week for news risk flagging
        try:
            import pandas as pd
            ts  = pd.Timestamp(state.get('timestamp', ''))
            dow = ts.day_name()
        except Exception:
            dow = 'Unknown'

        user_msg = f"""\
Kowalski, run the numbers on this trade.

SKIPPER'S ANALYSIS:
  Structure Quality : {ma.get('structure_quality', 'unknown')}
  Confluence Score  : {ma.get('confluence_score', 0)}/10
  Entry Zone Valid  : {ma.get('entry_zone_valid', False)}
  HTF Confluence    : {ma.get('htf_confluence', False)}
  Recommendation    : {ma.get('recommendation', 'hold')}
  ICT Pattern       : {ma.get('ict_pattern', 'none')}
  Skipper's Take    : "{ma.get('reasoning', 'N/A')}"

SIGNAL OVERVIEW:
{self._format_signal_context(state)}

RISK PARAMETERS:
  ADX            : {ctx.get('adx', 0):.1f}  ({'TRENDING ✅' if ctx.get('adx', 0) >= 25 else 'RANGING ⚠️'})
  ATR Ratio      : {ctx.get('atr_ratio', 1.0):.2f}
  Volatility     : {ctx.get('volatility', 'normal').upper()}
  Silver Bullet  : {state.get('is_silver_bullet', False)}
  Kill Zone Hour : {hour:02d}:00 UTC
  Day of Week    : {dow}

ACCOUNT:
  Balance        : ${state.get('balance', 0):.2f}
  SL Distance    : {state.get('sl_distance', 0):.1f} price units
  TP Distance    : {state.get('tp_distance', 0):.1f} price units
  RR Ratio       : {state.get('tp_distance', 0) / max(state.get('sl_distance', 1), 0.01):.2f}
  ML Confidence  : {state.get('confidence', 0.5):.2%}

Kowalski, is this mathematically sound? Should we proceed?"""

        # Append live news / calendar block
        if _HAS_NEWS:
            try:
                news_block = _get_news().get_summary(
                    symbols=Config.SYMBOLS, hours_ahead=6)
                user_msg += f"\n\nLIVE MARKET CONTEXT:\n{news_block}"
            except Exception as _ne:
                logging.debug(f"[Kowalski] news fetch: {_ne}")

        response = self._get_client().messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        return response.content[0].text

    # ── State update ──────────────────────────────────────────

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        approved      = result.get('approved', False)
        needs_news    = result.get('needs_news_check', False)
        # Also honour Skipper's news flag
        if (state.get('market_analysis') or {}).get('needs_news_check', False):
            result['needs_news_check'] = True
            needs_news = True

        logging.info(
            f"[Kowalski] 🧠 "
            f"Approved={approved} | "
            f"Risk={result.get('risk_level')} | "
            f"RR_Valid={result.get('rr_valid')} | "
            f"NewsCheck={needs_news} | "
            f"Reasons={result.get('rejection_reasons', [])} | "
            f"\"{result.get('reasoning', '')}\""
        )
        return {'risk_assessment': result}

    # ── Fallback ──────────────────────────────────────────────

    def _fallback(self, state: Dict) -> Dict:
        """Hard-coded risk rules when Claude is unavailable."""
        ctx      = state.get('context') or {}
        ma       = state.get('market_analysis') or {}
        hour     = state.get('hour', 0)
        adx      = ctx.get('adx', 0)
        is_sb    = state.get('is_silver_bullet', False)
        htf      = state.get('htf_bias', 'neutral')
        sig_type = (state.get('signal') or {}).get('type', 'unknown')
        balance  = state.get('balance', 200)
        atr_ratio = ctx.get('atr_ratio', 1.0)
        confluence = ma.get('confluence_score', 0)
        rec        = ma.get('recommendation', 'hold')

        reasons: List[str] = []
        approved = True

        if rec == 'hold':
            reasons.append("Skipper recommended HOLD")
            approved = False
        if adx < 25 and not is_sb:
            reasons.append(f"ADX {adx:.1f} < 25 outside SB window (R1)")
            approved = False
        if confluence < 6:
            reasons.append(f"Confluence {confluence}/10 < 6 (R2)")
            approved = False
        if htf not in ('neutral',):
            if (sig_type == 'buy' and htf == 'bearish') or \
               (sig_type == 'sell' and htf == 'bullish'):
                if ma.get('structure_quality') != 'strong':
                    reasons.append(f"HTF conflict: {htf} vs {sig_type} (R3)")
                    approved = False
        in_kz = hour in [13, 14, 15, 19, 20] or is_sb
        if not in_kz:
            reasons.append(f"Hour {hour:02d}:00 outside kill zones (R4)")
            approved = False
        if balance < 30:
            reasons.append(f"Balance ${balance:.2f} critically low (R6)")
            approved = False

        needs_news = atr_ratio > 1.8 or hour in [14, 15, 19]

        return {
            'risk_assessment': {
                'approved':          approved,
                'rejection_reasons': reasons,
                'risk_level':        'low' if approved and confluence >= 7
                                     else 'medium' if approved else 'high',
                'position_size_adj': 0.5 if atr_ratio > 2.0 else 1.0,
                'max_risk_percent':  0.005,
                'needs_news_check':  needs_news,
                'warnings':          [],
                'rr_valid':          True,
                'reasoning':         'Claude API unavailable — rule-based risk fallback',
            }
        }
