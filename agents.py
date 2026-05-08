"""
agents.py — Claude Single-Agent Trading Advisor
════════════════════════════════════════════════
One Claude call with three roles:
  1. ICT Analyst   — market structure analysis
  2. Risk Manager  — rule checking
  3. Lead Trader   — final decision

Called only when boti9 signal quality >= AGENT_MIN_QUALITY
so not on every candle, only strong signals.

Install:
    pip install anthropic

Add to Config:
    AGENT_ENABLED     = False   # True = enabled
    AGENT_MIN_QUALITY = 3.0     # call only on strong signals
    AGENT_API_KEY     = ""  # set via ANTHROPIC_API_KEY in ~/.penguin_squad/.env
"""

import json
import logging
import time
from typing import Dict, Optional

# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPT — three experts in one prompt
# ══════════════════════════════════════════════════════════════

TRADING_SYSTEM_PROMPT = """You are a trading committee of 3 experts analyzing NAS100 (USTECm) M15 trades.

EXPERT 1 — ICT ANALYST:
- Evaluate market structure quality (BOS vs CHoCH, premium/discount zones)
- Check if FVG/OB/Breaker zone is valid for entry
- Assess HTF alignment (H1 bias matches trade direction?)
- Rate signal confluence: 0-10

EXPERT 2 — RISK MANAGER (strict rules, protect $200 account):
- REJECT if ADX < 25 (ranging market, no trend)
- REJECT if signal_confluence < 6
- REJECT if trade direction conflicts with H1 bias
- REJECT if current hour is NOT in kill zones (13-16 UTC or 19-20 UTC Silver Bullet)
- REJECT if Vol=HIGH and no strong BOS confirmation
- APPROVE only if all rules pass

EXPERT 3 — LEAD TRADER (final decision):
- Weigh ICT analysis and Risk Manager verdict
- If Risk Manager rejects → action=skip (always)
- If approved → confirm direction (buy/sell) based on confluence
- Set confidence 0-100

CRITICAL RULES:
- Never override Risk Manager rejection
- For $200 account: when uncertain → skip
- BOS > CHoCH always
- Silver Bullet (15:00-16:00 UTC, 19:00-20:00 UTC) = higher priority

Respond ONLY in this exact JSON format, no other text:
{
  "ict_analysis": {
    "structure_quality": "strong|moderate|weak",
    "zone_valid": true/false,
    "htf_aligned": true/false,
    "confluence_score": 0-10,
    "key_observation": "one sentence"
  },
  "risk_verdict": {
    "approved": true/false,
    "rejection_reasons": [],
    "risk_level": "low|medium|high"
  },
  "lead_decision": {
    "action": "buy|sell|skip",
    "confidence": 0-100,
    "reasoning": "one sentence max"
  }
}"""


# ══════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ══════════════════════════════════════════════════════════════

def consult_agent(market_data: Dict, api_key: str) -> Optional[Dict]:
    """
    Call Claude to evaluate a trading signal.

    Args:
        market_data: dict with current market context
        api_key: Anthropic API key

    Returns:
        dict with agent decision, or None if call failed
    """
    try:
        import anthropic
    except ImportError:
        logging.error("[AGENT] ❌ anthropic not installed. Run: pip install anthropic")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # ── Build user message ────────────────────────────────────
    direction = market_data.get('signal_direction', 'unknown')
    quality   = market_data.get('signal_quality', 0)
    hour      = market_data.get('hour', 0)

    # Determine session phase for context
    if hour in [8, 9]:
        session_phase = "London Open"
    elif hour in [13, 14, 15]:
        session_phase = "NY Kill Zone"
    elif hour in [15, 16]:
        session_phase = "NY AM Silver Bullet"
    elif hour in [19, 20]:
        session_phase = "NY PM Silver Bullet"
    else:
        session_phase = "Regular Hours"

    user_message = f"""Analyze this {direction.upper()} signal on USTECm M15:

MARKET CONTEXT:
- Price: {market_data.get('price', 0):.2f}
- Signal Direction: {direction.upper()}
- Signal Quality Score: {quality:.1f}/10
- Session: {market_data.get('session', 'new_york')} | Phase: {session_phase} ({hour}:00 UTC)

TECHNICAL DATA:
- ADX: {market_data.get('adx', 0):.1f} ({'TRENDING' if market_data.get('adx', 0) >= 25 else 'RANGING'})
- ATR: {market_data.get('atr', 0):.2f} | ATR Ratio: {market_data.get('atr_ratio', 1.0):.2f}
- Volatility: {market_data.get('volatility', 'NORMAL')}
- RSI: {market_data.get('rsi', 50):.1f}

ICT STRUCTURE:
- Market Structure: {market_data.get('structure', 'no_structure')}
- HTF Bias (H1): {market_data.get('htf_bias', 'neutral')}
- HTF Aligned: {market_data.get('htf_aligned', False)}
- FVG Present: {market_data.get('fvg_present', False)}
- Zone Type: {market_data.get('zone_type', 'equilibrium')} (premium/discount/equilibrium)
- Silver Bullet Time: {market_data.get('is_silver_bullet', False)}
- Liquidity Swept: {market_data.get('liquidity_swept', False)}

ML MODEL:
- ML Confidence: {market_data.get('ml_confidence', 0.5):.2%}
- Pattern: {market_data.get('pattern_key', 'unknown')}

PROPOSED TRADE:
- Entry: {market_data.get('entry_price', 0):.2f}
- Stop Loss: {market_data.get('sl_price', 0):.2f} ({market_data.get('sl_distance', 0):.1f} units away)
- Take Profit: {market_data.get('tp_price', 0):.2f} ({market_data.get('tp_distance', 0):.1f} units away)
- Lot Size: {market_data.get('lot_size', 0.01):.2f}
- Account Balance: ${market_data.get('balance', 200):.2f}

Should we take this trade?"""

    # ── API Call ──────────────────────────────────────────────
    start_time = time.time()
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=TRADING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        elapsed = time.time() - start_time
        raw_text = response.content[0].text.strip()

        logging.info(f"[AGENT] ✅ Response received in {elapsed:.1f}s "
                     f"(tokens: {response.usage.input_tokens}in + {response.usage.output_tokens}out)")

        # ── Parse JSON ────────────────────────────────────────
        # Clean markdown if present
        if "```" in raw_text:
            parts = raw_text.split("```")
            for part in parts:
                if "{" in part:
                    raw_text = part.replace("json", "").strip()
                    break

        result = json.loads(raw_text)

        # ── Log decision ──────────────────────────────────────
        ict    = result.get("ict_analysis", {})
        risk   = result.get("risk_verdict", {})
        lead   = result.get("lead_decision", {})

        action     = lead.get("action", "skip")
        confidence = lead.get("confidence", 0)
        approved   = risk.get("approved", False)

        log_emoji = "✅" if action != "skip" else "⏭️"
        logging.info(
            f"[AGENT] {log_emoji} Decision: {action.upper()} "
            f"(confidence={confidence}%, approved={approved})\n"
            f"  ICT: confluence={ict.get('confluence_score')}/10, "
            f"zone={'valid' if ict.get('zone_valid') else 'invalid'}, "
            f"htf={'aligned' if ict.get('htf_aligned') else 'conflict'}\n"
            f"  Risk: {risk.get('risk_level')} | "
            f"Rejections: {risk.get('rejection_reasons', [])}\n"
            f"  Reason: {lead.get('reasoning', '')}"
        )

        return result

    except json.JSONDecodeError as e:
        logging.error(f"[AGENT] ❌ JSON parse failed: {e}\nRaw: {raw_text[:200]}")
        return None
    except Exception as e:
        logging.error(f"[AGENT] ❌ API call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# DECISION EXTRACTOR
# ══════════════════════════════════════════════════════════════

def get_agent_action(agent_result: Optional[Dict]) -> str:
    """
    Extract the action from the agent result.
    Returns: 'buy', 'sell', or 'skip'
    """
    if agent_result is None:
        return "skip"  # API failed → skip for safety

    risk = agent_result.get("risk_verdict", {})
    lead = agent_result.get("lead_decision", {})

    # Risk Manager rejection is absolute
    if not risk.get("approved", False):
        return "skip"

    action = lead.get("action", "skip")
    confidence = lead.get("confidence", 0)

    # Low confidence → skip
    if confidence < 55:
        logging.info(f"[AGENT] ⏭️ Confidence {confidence}% < 55% threshold — skipping")
        return "skip"

    return action if action in ["buy", "sell"] else "skip"


# ══════════════════════════════════════════════════════════════
# COST TRACKER
# ══════════════════════════════════════════════════════════════

class AgentCostTracker:
    """API cost tracker"""

    def __init__(self):
        self.total_calls = 0
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        # Claude Sonnet 4.6 pricing: $3/1M input, $15/1M output
        self.input_price_per_1m  = 3.0
        self.output_price_per_1m = 15.0

    def record(self, input_tokens: int, output_tokens: int):
        self.total_calls += 1
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens

    @property
    def total_cost_usd(self) -> float:
        input_cost  = (self.total_input_tokens  / 1_000_000) * self.input_price_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_price_per_1m
        return input_cost + output_cost

    def log_summary(self):
        logging.info(
            f"[AGENT_COST] Calls={self.total_calls}, "
            f"Tokens={self.total_input_tokens}in+{self.total_output_tokens}out, "
            f"Cost=${self.total_cost_usd:.4f}"
        )


# Global cost tracker instance
cost_tracker = AgentCostTracker()
