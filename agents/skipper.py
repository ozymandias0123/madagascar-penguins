"""
agents/skipper.py — Skipper (Market Analyst)
Model  : GPT-4o
Role   : Flat-headed decisive leader. First to analyse the market using ICT
         concepts — FVG, BOS, CHoCH, Silver Bullet, Order Blocks.
         Issues a clear directive: BUY / SELL / HOLD.
Node   : skipper_node  →  updates state['market_analysis']
"""

import logging
from typing import Dict

from config import Config
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """\
You are Skipper — the decisive, flat-headed leader of the trading team.
You analyse NAS100/USTECm M15 charts using pure ICT (Inner Circle Trader) methodology.

YOUR FRAMEWORK:
• BOS (Break of Structure) > CHoCH — BOS = stronger signal, always prioritise it
• Silver Bullet windows (London 08-09 UTC, NY AM 15-16 UTC, NY PM 19-20 UTC) = top priority
• FVG (Fair Value Gap), Order Blocks, and Breaker Blocks define valid entry zones
• Premium zone (price > 70 % range) = look for SELLS only
• Discount zone (price < 30 % range) = look for BUYS only
• HTF bias alignment is NON-NEGOTIABLE
• ADX > 25 confirms trend strength — below 25 in a ranging market = HOLD

YOUR PERSONALITY:
• Decisive — never say "maybe". Give a clear directive.
• Strict — only A-grade setups meet your standard.
• Concise — one sharp observation, no waffle.

RESPOND ONLY with this exact JSON (no markdown, no extra text):
{
  "trend_direction":    "bullish|bearish|neutral",
  "structure_quality":  "strong|moderate|weak",
  "key_levels": {
    "support":    0.0,
    "resistance": 0.0
  },
  "htf_confluence":    true,
  "ict_pattern":       "bos|choch|fvg|ob|breaker|none",
  "entry_zone_valid":  true,
  "confluence_score":  7,
  "recommendation":    "buy|sell|hold",
  "needs_news_check":  false,
  "reasoning":         "one sharp sentence — Skipper style"
}

NOTE: set needs_news_check=true if you detect elevated volatility, a
potential news-driven move, or if macro context is ambiguous."""


class Skipper(BaseAgent):
    """
    Skipper — ICT Market Analyst.
    Provider is chosen at setup: 'openai' (ChatGPT) or 'gemini'.
    """
    name = "Skipper"

    def __init__(self):
        self._provider = Config.SKIPPER_PROVIDER   # 'openai' or 'gemini'
        self._client   = None
        self._model    = None

        if self._provider == 'gemini':
            if not Config.GOOGLE_API_KEY:
                logging.warning("[Skipper] GOOGLE_API_KEY not set — will use fallback")
        else:
            if not Config.OPENAI_API_KEY:
                logging.warning("[Skipper] OPENAI_API_KEY not set — will use fallback")

        logging.info(f"[Skipper] Provider: {self._provider.upper()}")

    def _get_openai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        return self._client

    def _get_gemini(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self._model = genai.GenerativeModel(Config.GEMINI_MODEL)
        return self._model

    # ── LLM call ─────────────────────────────────────────────

    def _call_llm(self, state: Dict) -> str:
        sig     = state.get('signal') or {}
        ctx     = state.get('context') or {}
        df      = state.get('df')
        rsi_val = 50.0
        if df is not None and 'rsi' in df.columns and len(df) > 2:
            rsi_val = float(df['rsi'].iloc[-2])

        direction = self._direction_from_signal(state)
        user_msg  = f"""\
Skipper, analyse this {direction.upper()} signal — give me your verdict!

MARKET SNAPSHOT:
{self._format_signal_context(state)}

TECHNICAL DETAIL:
- RSI:              {rsi_val:.1f}
- ATR:              {state.get('atr', 0):.2f}  |  ATR Ratio: {ctx.get('atr_ratio', 1.0):.2f}
- ML Confidence:    {state.get('confidence', 0.5):.2%}
- Zone Type:        {state.get('zone_type', 'equilibrium')}
- FVG Present:      {len(state.get('fvg_zones', [])) > 0}
- Liquidity Swept:  {state.get('liquidity_swept', False)}
- Silver Bullet:    {state.get('is_silver_bullet', False)}
- Entry:            {state.get('entry_price', 0):.2f}
- SL:               {state.get('sl_price', 0):.2f}  ({state.get('sl_distance', 0):.1f} units)
- TP:               {state.get('tp_price', 0):.2f}  ({state.get('tp_distance', 0):.1f} units)
- RR Ratio:         {state.get('tp_distance', 0) / max(state.get('sl_distance', 1), 0.01):.1f}
- ICT Pattern:      {sig.get('pattern_key', 'unknown')}

What's the play, Skipper?"""

        if self._provider == 'gemini':
            if not Config.GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY missing")
            response = self._get_gemini().generate_content(
                f"{SYSTEM_PROMPT}\n\n{user_msg}",
                generation_config={"temperature": 0.15, "max_output_tokens": 400},
            )
            return response.text
        else:
            if not Config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY missing")
            response = self._get_openai().chat.completions.create(
                model=Config.OPENAI_MODEL,
                max_tokens=400,
                temperature=0.15,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ]
            )
            return response.choices[0].message.content

    # ── State update ──────────────────────────────────────────

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        logging.info(
            f"[Skipper] 🐧 "
            f"Structure={result.get('structure_quality')} | "
            f"Confluence={result.get('confluence_score')}/10 | "
            f"Rec={result.get('recommendation')} | "
            f"NewsCheck={result.get('needs_news_check', False)} | "
            f"\"{result.get('reasoning', '')}\""
        )
        return {'market_analysis': result}

    # ── Fallback ──────────────────────────────────────────────

    def _fallback(self, state: Dict) -> Dict:
        direction = self._direction_from_signal(state)
        ctx       = state.get('context') or {}
        adx       = ctx.get('adx', 0)
        structure = state.get('structure', '')
        is_sb     = state.get('is_silver_bullet', False)

        # Rule-based quality estimate
        if 'bos' in structure and adx >= 25:
            quality, rec = 7, direction
        elif 'bos' in structure:
            quality, rec = 5, direction
        elif 'choch' in structure:
            quality, rec = 4, direction
        else:
            quality, rec = 2, 'hold'

        return {
            'market_analysis': {
                'trend_direction':    direction if rec != 'hold' else 'neutral',
                'structure_quality':  'strong' if quality >= 7 else
                                      'moderate' if quality >= 5 else 'weak',
                'key_levels':         {'support': 0.0, 'resistance': 0.0},
                'htf_confluence':     state.get('htf_bias') != 'neutral',
                'ict_pattern':        'bos'   if 'bos'   in structure else
                                      'choch' if 'choch' in structure else 'none',
                'entry_zone_valid':   quality >= 5,
                'confluence_score':   quality,
                'recommendation':     rec,
                'needs_news_check':   False,
                'reasoning':          'GPT-4o API unavailable — rule-based fallback',
            }
        }
