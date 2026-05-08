"""
agents/validator.py — DeepseekValidator
Uses DeepSeek (OpenAI-compatible API) as an independent second opinion
that synthesises ALL previous analyses into a final go/no-go verdict.
Node: validator_node  →  updates state['validation'], state['approved'],
                         state['final_action'], state['rejection_reasons']
"""

import logging
from typing import Dict, List

from config import Config
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """You are DeepSeek Validator — the final decision-maker in a multi-agent \
NAS100 trading system.

You receive analyses from three specialists:
  1. Skifer (ICT Market Analyst, GPT-4o)
  2. Gemini (News/Macro Analyst)
  3. Kovalski (Risk Manager, Claude)

YOUR ROLE:
- Synthesise all three analyses.
- If Kovalski (Risk Manager) rejected → you MUST also reject. No exceptions.
- If all three agree → follow consensus.
- If mixed signals → apply conservative bias (lean toward SKIP).
- Confidence threshold: only approve if confidence >= 55%.

DECISION CRITERIA:
- approved = true  ONLY if: Risk Manager approved AND confluence >= 6 AND confidence >= 55
- final_action = "buy" | "sell" only if approved = true
- final_action = "skip" if approved = false

RESPOND ONLY with this exact JSON — no other text:
{
  "approved":              false,
  "confidence":            0,
  "final_direction":       "buy|sell|skip",
  "risk_reward_acceptable": false,
  "agent_consensus":       "agree|mixed|disagree",
  "rejection_reasons":     [],
  "reasoning":             "one concise sentence"
}"""


class DeepseekValidator(BaseAgent):
    name = "DeepseekValidator"

    def __init__(self):
        if not Config.DEEPSEEK_API_KEY:
            logging.warning("[DeepseekValidator] DEEPSEEK_API_KEY not set — will use fallback")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url="https://api.deepseek.com"
            )
        return self._client

    # ── LLM call ─────────────────────────────────────────────

    def _call_llm(self, state: Dict) -> str:
        if not Config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY missing")

        client = self._get_client()
        ma     = state.get('market_analysis') or {}
        na     = state.get('news_analysis')   or {}
        ra     = state.get('risk_assessment') or {}

        user_msg = f"""Validate this trade — synthesise all specialist analyses:

ORIGINAL SIGNAL:
{self._format_signal_context(state)}

═══════════════════════════════════════════
SKIFER — ICT Market Analyst (GPT-4o):
- Structure Quality:  {ma.get('structure_quality', 'unknown')}
- Confluence Score:   {ma.get('confluence_score', 0)}/10
- Entry Zone Valid:   {ma.get('entry_zone_valid', False)}
- HTF Confluence:     {ma.get('htf_confluence', False)}
- Recommendation:     {ma.get('recommendation', 'wait')}
- Reasoning:          {ma.get('reasoning', 'N/A')}

═══════════════════════════════════════════
GEMINI — News/Macro Analyst:
- Session Sentiment:  {na.get('session_sentiment', 'neutral')}
- Macro Risk Level:   {na.get('macro_risk_level', 'medium')}
- Safe to Trade:      {na.get('safe_to_trade', True)}
- Session Quality:    {na.get('session_quality', 'standard')}
- Notes:              {na.get('notes', 'N/A')}

═══════════════════════════════════════════
KOVALSKI — Risk Manager (Claude):
- APPROVED:           {ra.get('approved', False)}
- Risk Level:         {ra.get('risk_level', 'high')}
- Rejection Reasons:  {ra.get('rejection_reasons', [])}
- Warnings:           {ra.get('warnings', [])}
- Reasoning:          {ra.get('reasoning', 'N/A')}

═══════════════════════════════════════════
ML MODEL:
- Confidence:         {state.get('confidence', 0.5):.2%}
- Quality Score:      {state.get('quality_score', 0):.1f}

Make your final validation decision."""

        response = self._get_client().chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            max_tokens=400,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ]
        )
        return response.choices[0].message.content

    # ── State update ──────────────────────────────────────────

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        approved    = result.get('approved', False)
        confidence  = result.get('confidence', 0)
        direction   = result.get('final_direction', 'skip')
        reasons     = result.get('rejection_reasons', [])

        # Enforce confidence threshold
        if confidence < Config.AGENT_MIN_CONFIDENCE:
            approved  = False
            direction = 'skip'
            reasons.append(f"Confidence {confidence}% < {Config.AGENT_MIN_CONFIDENCE}% threshold")

        # Risk Manager veto is absolute
        ra = state.get('risk_assessment') or {}
        if not ra.get('approved', False):
            approved  = False
            direction = 'skip'
            if 'Risk Manager rejected' not in str(reasons):
                reasons.extend(ra.get('rejection_reasons', ['Risk Manager rejected']))

        final_action = direction if approved and direction in ('buy', 'sell') else 'skip'

        logging.info(
            f"[DeepseekValidator] ✔️ "
            f"Approved={approved} | Confidence={confidence}% | "
            f"Action={final_action} | Consensus={result.get('agent_consensus')} | "
            f"{result.get('reasoning', '')}"
        )

        return {
            'validation':       result,
            'approved':         approved,
            'final_action':     final_action,
            'rejection_reasons': reasons,
        }

    def _fallback(self, state: Dict) -> Dict:
        """
        Fallback: trust Kovalski's verdict when DeepSeek is unavailable.
        If Risk Manager approved → pass through; otherwise reject.
        """
        ra         = state.get('risk_assessment') or {}
        rm_approved = ra.get('approved', False)
        sig_dir    = (state.get('signal') or {}).get('type', 'skip')
        confidence  = int(state.get('confidence', 0.5) * 100)
        reasons: List[str] = []

        if not rm_approved:
            reasons = ra.get('rejection_reasons', ['Risk Manager rejected'])
            action  = 'skip'
        elif confidence < Config.AGENT_MIN_CONFIDENCE:
            reasons = [f"ML confidence {confidence}% below threshold"]
            action  = 'skip'
            rm_approved = False
        else:
            action = sig_dir if sig_dir in ('buy', 'sell') else 'skip'
            if action == 'skip':
                rm_approved = False

        return {
            'validation': {
                'approved':              rm_approved,
                'confidence':            confidence,
                'final_direction':       action,
                'risk_reward_acceptable': rm_approved,
                'agent_consensus':       'mixed',
                'rejection_reasons':     reasons,
                'reasoning':             'DeepSeek API unavailable — using Risk Manager verdict'
            },
            'approved':          rm_approved,
            'final_action':      action,
            'rejection_reasons': reasons,
        }
