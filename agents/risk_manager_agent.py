"""
agents/risk_manager_agent.py — KovalksiRiskManager
Uses Claude to enforce strict risk rules and issue a go/no-go verdict.
Node: risk_manager_node  →  updates state['risk_assessment']

This is the most conservative agent — its REJECTION is ABSOLUTE.
"""

import logging
from typing import Dict

from config import Config
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are Kovalski, a strict risk manager protecting a small trading account.

YOUR ABSOLUTE REJECTION RULES — if ANY applies, set approved=false:
1. ADX < 25 AND NOT in a Silver Bullet window
2. signal_confluence_score < 6
3. Trade direction conflicts with H1 HTF bias (unless structure is very strong BOS)
4. Current hour is NOT in kill zones: 13-16 UTC or 19-20 UTC (Silver Bullet)
5. Volatility=HIGH with no BOS confirmation (CHoCH only is insufficient)
6. Macro risk level = 'high' (from News Analyst)
7. Account balance is dangerously low (< $50)
8. session_quality = 'low' from News Analyst AND confluence_score < 7

APPROVAL CRITERIA — ALL must pass:
- ADX >= 25 OR Silver Bullet window with ADX >= 15
- HTF bias aligned OR strong BOS structure
- In a valid session kill zone
- News Analyst says safe_to_trade = true
- ML confidence >= 40%

RISK LEVELS:
- low:    All criteria pass with strong confluence (>= 7/10)
- medium: Most criteria pass, some marginal values
- high:   Borderline case — approved but with warnings

RESPOND ONLY with this exact JSON — no other text:
{
  "approved":          false,
  "rejection_reasons": [],
  "risk_level":        "low|medium|high",
  "position_size_adj": 1.0,
  "max_risk_percent":  0.005,
  "warnings":          [],
  "reasoning":         "one concise sentence"
}"""


class KovalksiRiskManager(BaseAgent):
    name = "KovalksiRiskManager"

    def __init__(self):
        if not Config.ANTHROPIC_API_KEY:
            logging.warning("[KovalksiRiskManager] ANTHROPIC_API_KEY not set — will use fallback")
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

        client = self._get_client()
        ctx    = state.get('context') or {}
        ma     = state.get('market_analysis') or {}
        na     = state.get('news_analysis')   or {}
        hour   = state.get('hour', 0)

        user_msg = f"""Review this trade for risk approval:

SIGNAL OVERVIEW:
{self._format_signal_context(state)}

MARKET ANALYST (Skifer) ASSESSMENT:
- Structure Quality:  {ma.get('structure_quality', 'unknown')}
- Confluence Score:   {ma.get('confluence_score', 0)}/10
- Entry Zone Valid:   {ma.get('entry_zone_valid', False)}
- HTF Confluence:     {ma.get('htf_confluence', False)}
- Recommendation:     {ma.get('recommendation', 'wait')}
- ICT Pattern:        {ma.get('ict_pattern', 'none')}

NEWS ANALYST (Gemini) ASSESSMENT:
- Session Sentiment:  {na.get('session_sentiment', 'neutral')}
- Macro Risk Level:   {na.get('macro_risk_level', 'medium')}
- Safe to Trade:      {na.get('safe_to_trade', True)}
- Session Quality:    {na.get('session_quality', 'standard')}
- News Event Risk:    {na.get('news_event_risk', False)}

TECHNICAL DETAILS:
- ADX:              {ctx.get('adx', 0):.1f} ({'TRENDING' if ctx.get('adx', 0) >= 25 else 'RANGING'})
- ATR Ratio:        {ctx.get('atr_ratio', 1.0):.2f}
- Volatility:       {ctx.get('volatility', 'normal').upper()}
- ML Confidence:    {state.get('confidence', 0.5):.2%}
- Silver Bullet:    {state.get('is_silver_bullet', False)}
- Kill Zone Hour:   {hour:02d}:00 UTC

ACCOUNT:
- Balance:          ${state.get('balance', 0):.2f}
- Quality Score:    {state.get('quality_score', 0):.1f}
- SL Distance:      {state.get('sl_distance', 0):.1f} price units
- TP Distance:      {state.get('tp_distance', 0):.1f} price units

Should this trade be approved? Apply your strict risk rules."""

        response = client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        return response.content[0].text

    # ── State update ──────────────────────────────────────────

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        approved = result.get('approved', False)
        reasons  = result.get('rejection_reasons', [])
        logging.info(
            f"[KovalksiRiskManager] 🛡️ "
            f"Approved={approved} | "
            f"Risk={result.get('risk_level')} | "
            f"Reasons={reasons} | "
            f"{result.get('reasoning', '')}"
        )
        return {'risk_assessment': result}

    def _fallback(self, state: Dict) -> Dict:
        """
        Safe fallback: apply hard-coded rules when Claude is unavailable.
        This mirrors the original ADX + kill-zone + quality rules from ozy.py.
        """
        ctx    = state.get('context') or {}
        hour   = state.get('hour', 0)
        adx    = ctx.get('adx', 0)
        is_sb  = state.get('is_silver_bullet', False)
        quality = state.get('quality_score', 0)
        htf    = state.get('htf_bias', 'neutral')
        sig_type = (state.get('signal') or {}).get('type', 'unknown')

        reasons = []
        approved = True

        if adx < 25 and not is_sb:
            reasons.append(f"ADX {adx:.1f} < 25 outside SB window")
            approved = False
        if adx < 15:
            reasons.append(f"ADX {adx:.1f} critically low")
            approved = False
        in_kz = hour in [13, 14, 15, 19, 20] or is_sb
        if not in_kz:
            reasons.append(f"Hour {hour:02d} outside kill zones")
            approved = False
        if htf not in ('neutral',) and (
            (sig_type == 'buy'  and htf == 'bearish') or
            (sig_type == 'sell' and htf == 'bullish')
        ):
            reasons.append(f"HTF bias conflict: {htf} vs {sig_type}")
            approved = False
        if state.get('balance', 200) < 50:
            reasons.append(f"Balance too low: ${state.get('balance', 0):.2f}")
            approved = False

        return {
            'risk_assessment': {
                'approved':          approved,
                'rejection_reasons': reasons,
                'risk_level':        'low' if approved and quality >= 2.5 else
                                     'medium' if approved else 'high',
                'position_size_adj': 1.0,
                'max_risk_percent':  0.005,
                'warnings':          [],
                'reasoning':         'Claude API unavailable — rule-based fallback applied'
            }
        }
