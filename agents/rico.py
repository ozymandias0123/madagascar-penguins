"""
agents/rico.py — Rico (Final Validator)
Model  : DeepSeek-chat
Role   : The wild card who makes the FINAL go / no-go call.
         Reviews Skipper's market analysis, Kowalski's risk assessment,
         and all market context, then stamps the trade approved or rejected.
         Leans toward caution: when in doubt → skip.

Node   : rico_node  →  updates state['validation'], state['approved'],
                        state['final_action'], state['rejection_reasons']
"""

import logging
from typing import Dict, List

from config import Config
from agents.base_agent import BaseAgent


SYSTEM_PROMPT = """\
You are Rico — the wild card of the trading team. Unpredictable on the surface,
but underneath you make the FINAL decision whether to execute a trade or skip it.
You trust your teammates but you double-check everything yourself.

YOUR DECISION RULES:
1. If Kowalski (Risk Manager) REJECTED → you MUST also reject. No exceptions.
   "If Kowalski says no, that's a no."
2. Review ALL teammate inputs together. If they contradict → be conservative → skip.
3. Confidence gate: only approve if your own confidence >= 55 %.
4. When uncertain — skip. Protect the account first.

CONFIDENCE CALCULATION GUIDE:
  Start at 50. Then adjust:
  +15 if Kowalski approved AND risk_level = "low"
  +10 if Skipper confluence >= 8
  + 8 if Silver Bullet window active
  + 7 if HTF confluence = true
  + 5 if structure_quality = "strong"
  − 10 if risk_level = "high"
  − 20 if Kowalski rejected

AGENT CONSENSUS LEVELS:
  agree    — Skipper rec + Kowalski approved all align
  mixed    — some conflict between agents
  disagree — agents contradict each other

RESPOND ONLY with this exact JSON (no markdown, no extra text):
{
  "approved":               false,
  "confidence":             0,
  "final_direction":        "buy|sell|skip",
  "risk_reward_acceptable": false,
  "agent_consensus":        "agree|mixed|disagree",
  "rejection_reasons":      [],
  "reasoning":              "one Rico-style sentence"
}"""


class Rico(BaseAgent):
    """
    Rico — Final Validator powered by DeepSeek-chat.
    The wild card who makes the last, most important call.
    """
    name = "Rico"

    def __init__(self):
        if not Config.DEEPSEEK_API_KEY:
            logging.warning("[Rico] DEEPSEEK_API_KEY not set — will use fallback")
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

        ma = state.get('market_analysis') or {}
        ra = state.get('risk_assessment') or {}

        user_msg = f"""\
Rico, everyone's weighed in. Time for YOUR final decision.

═══ SKIPPER (GPT-4o) ════════════════════════════════
  Structure Quality : {ma.get('structure_quality', 'unknown')}
  Confluence Score  : {ma.get('confluence_score', 0)}/10
  Entry Zone Valid  : {ma.get('entry_zone_valid', False)}
  HTF Confluence    : {ma.get('htf_confluence', False)}
  Recommendation    : {ma.get('recommendation', 'hold')}
  Skipper's Words   : "{ma.get('reasoning', 'N/A')}"

═══ KOWALSKI (Claude) ═══════════════════════════════
  APPROVED          : {ra.get('approved', False)}
  Risk Level        : {ra.get('risk_level', 'high')}
  RR Valid          : {ra.get('rr_valid', False)}
  Rejection Reasons : {ra.get('rejection_reasons', [])}
  Warnings          : {ra.get('warnings', [])}
  Kowalski's Words  : "{ra.get('reasoning', 'N/A')}"

═══ SIGNAL SUMMARY ══════════════════════════════════
{self._format_signal_context(state)}

  ML Confidence     : {state.get('confidence', 0.5):.2%}
  Quality Score     : {state.get('quality_score', 0):.1f}
  Silver Bullet     : {state.get('is_silver_bullet', False)}
  Balance           : ${state.get('balance', 0):.2f}

Rico, is this trade worth it? Be careful. Real money is on the line."""

        response = self._get_client().chat.completions.create(
            model=Config.DEEPSEEK_MODEL,
            max_tokens=400,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]
        )
        return response.choices[0].message.content

    # ── State update ──────────────────────────────────────────

    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        approved   = result.get('approved', False)
        confidence = result.get('confidence', 0)
        direction  = result.get('final_direction', 'skip')
        reasons    = result.get('rejection_reasons', [])

        # Confidence gate
        if confidence < Config.AGENT_MIN_CONFIDENCE:
            approved  = False
            direction = 'skip'
            reasons.append(
                f"Rico: confidence {confidence}% < {Config.AGENT_MIN_CONFIDENCE}% gate"
            )

        # Kowalski's veto is absolute — Rico always enforces it
        ra = state.get('risk_assessment') or {}
        if not ra.get('approved', False):
            approved  = False
            direction = 'skip'
            rm_reasons = ra.get('rejection_reasons', ['Kowalski rejected'])
            for r in rm_reasons:
                if r not in reasons:
                    reasons.append(r)

        final_action = direction if approved and direction in ('buy', 'sell') else 'skip'
        if final_action == 'skip':
            approved = False

        logging.info(
            f"[Rico] 🃏 "
            f"Approved={approved} | Confidence={confidence}% | "
            f"Action={final_action} | Consensus={result.get('agent_consensus')} | "
            f"\"{result.get('reasoning', '')}\""
        )

        return {
            'validation':        result,
            'approved':          approved,
            'final_action':      final_action,
            'rejection_reasons': reasons,
        }

    # ── Fallback ──────────────────────────────────────────────

    def _fallback(self, state: Dict) -> Dict:
        """Fallback: trust Kowalski's verdict if DeepSeek is unavailable."""
        ra          = state.get('risk_assessment') or {}
        rm_approved = ra.get('approved', False)
        sig_dir     = (state.get('signal') or {}).get('type', 'skip')
        confidence  = int(state.get('confidence', 0.5) * 100)
        reasons: List[str] = []

        if not rm_approved:
            reasons  = ra.get('rejection_reasons', ['Kowalski rejected'])
            action   = 'skip'
            approved = False
        elif confidence < Config.AGENT_MIN_CONFIDENCE:
            reasons  = [f"ML confidence {confidence}% below {Config.AGENT_MIN_CONFIDENCE}% gate"]
            action   = 'skip'
            approved = False
        else:
            action   = sig_dir if sig_dir in ('buy', 'sell') else 'skip'
            approved = action in ('buy', 'sell')

        return {
            'validation': {
                'approved':               approved,
                'confidence':             confidence,
                'final_direction':        action,
                'risk_reward_acceptable': approved,
                'agent_consensus':        'mixed',
                'rejection_reasons':      reasons,
                'reasoning':              'DeepSeek unavailable — deferring to Kowalski',
            },
            'approved':          approved,
            'final_action':      action,
            'rejection_reasons': reasons,
        }
