"""
agents/market_analyst.py — SkiferMarketAnalyst
Uses GPT-4o to perform deep ICT technical analysis.
Node: market_analyst_node  →  updates state['market_analysis']
"""

import logging
from typing import Dict

from config import Config
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are Skifer, an elite ICT (Inner Circle Trader) technical analyst \
specialising in NAS100 / USTECm M15 futures.

Your job is to analyse the provided market snapshot and give a comprehensive ICT assessment.

FRAMEWORK:
- BOS (Break of Structure) > CHoCH (Change of Character) in signal strength
- Silver Bullet windows (London 08-09, NY 15-16, 19-20 UTC) are high-priority
- FVG, Order Blocks, and Breaker Blocks define entry zones
- Premium zones → look for sells; Discount zones → look for buys
- HTF bias alignment is mandatory for high-confidence trades
- ADX > 25 confirms trend; ADX < 25 = ranging (extra caution)

RESPOND ONLY with this exact JSON — no other text:
{
  "trend_direction":   "bullish|bearish|neutral",
  "structure_quality": "strong|moderate|weak",
  "key_levels": {
    "support":    0.0,
    "resistance": 0.0
  },
  "htf_confluence":    true,
  "ict_pattern":       "bos|choch|fvg|ob|breaker|none",
  "entry_zone_valid":  true,
  "confluence_score":  7,
  "recommendation":    "buy|sell|wait",
  "reasoning":         "one concise sentence"
}"""


class SkiferMarketAnalyst(BaseAgent):
    name = "SkiferMarketAnalyst"

    def __init__(self):
        if not Config.OPENAI_API_KEY:
            logging.warning("[SkiferMarketAnalyst] OPENAI_API_KEY not set — will use fallback")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=Config.OPENAI_API_KEY)
        return self._client

    # ── LLM call ─────────────────────────────────────────────

    def _call_llm(self, state: Dict) -> str:
        if not Config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY missing")

        client  = self._get_client()
        sig     = state.get('signal') or {}
        ctx     = state.get('context') or {}
        df      = state.get('df')
        rsi_val = 50.0
        if df is not None and 'rsi' in df.columns and len(df) > 2:
            rsi_val = float(df['rsi'].iloc[-2])

        user_msg = f"""Analyse this {self._direction_from_signal(state).upper()} trade signal:

MARKET SNAPSHOT:
{self._format_signal_context(state)}

TECHNICAL DETAILS:
- RSI: {rsi_val:.1f}
- ATR: {state.get('atr', 0):.2f} | ATR Ratio: {ctx.get('atr_ratio', 1.0):.2f}
- ML Confidence: {state.get('confidence', 0.5):.2%}
- Zone Type: {state.get('zone_type', 'equilibrium')} (premium/discount/equilibrium)
- FVG Present: {len(state.get('fvg_zones', [])) > 0}
- Liquidity Swept: {state.get('liquidity_swept', False)}
- Silver Bullet Window: {state.get('is_silver_bullet', False)}
- Entry: {state.get('entry_price', 0):.2f}
- SL Distance: {state.get('sl_distance', 0):.1f} price units
- TP Distance: {state.get('tp_distance', 0):.1f} price units (RR = {state.get('tp_distance', 0) / max(state.get('sl_distance', 1), 0.01):.1f})

PATTERN: {sig.get('pattern_key', 'unknown')}

Provide your ICT analysis as JSON."""

        response = client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            max_tokens=400,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ]
        )
        return response.choices[0].message.content

    # ── State update ──────────────────────────────────────────

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        logging.info(
            f"[SkiferMarketAnalyst] 📊 "
            f"Structure={result.get('structure_quality')} | "
            f"Confluence={result.get('confluence_score')}/10 | "
            f"Rec={result.get('recommendation')} | "
            f"{result.get('reasoning', '')}"
        )
        return {'market_analysis': result}

    def _fallback(self, state: Dict) -> Dict:
        """Safe fallback — neutral analysis, does NOT block trading."""
        direction = self._direction_from_signal(state)
        return {
            'market_analysis': {
                'trend_direction':   direction,
                'structure_quality': 'moderate',
                'key_levels':        {'support': 0.0, 'resistance': 0.0},
                'htf_confluence':    state.get('htf_bias') != 'neutral',
                'ict_pattern':       'bos' if 'bos' in state.get('structure', '') else 'none',
                'entry_zone_valid':  True,
                'confluence_score':  5,
                'recommendation':    direction if direction in ('buy', 'sell') else 'wait',
                'reasoning':         'API unavailable — using rule-based fallback'
            }
        }
