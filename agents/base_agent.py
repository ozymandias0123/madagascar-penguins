"""
agents/base_agent.py — Abstract base class for all trading agents.

Every agent:
  • Receives the full TradingState dict.
  • Returns a dict of ONLY the keys it wants to update in the state.
  • Handles its own API errors gracefully and returns a safe fallback.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Abstract base for all multi-agent nodes.

    Subclasses implement `_call_llm()` which must return a raw JSON string.
    `run()` handles parsing, retries, and fallback logic.
    """

    name: str = "BaseAgent"
    max_retries: int = 2
    retry_delay: float = 2.0

    # ── Public interface ──────────────────────────────────────

    def run(self, state: Dict) -> Dict:
        """
        Execute the agent and return a partial state update dict.
        Never raises — always returns something safe.
        """
        for attempt in range(self.max_retries):
            try:
                raw = self._call_llm(state)
                result = self._parse_json(raw)
                if result:
                    logging.info(
                        f"[{self.name}] ✅ Response received "
                        f"(attempt {attempt + 1})"
                    )
                    return self._build_state_update(result, state)
            except Exception as exc:
                logging.warning(
                    f"[{self.name}] ⚠️ Attempt {attempt + 1} failed: {exc}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        logging.error(f"[{self.name}] ❌ All retries exhausted — using fallback")
        return self._fallback(state)

    # ── Abstract methods (must implement) ─────────────────────

    @abstractmethod
    def _call_llm(self, state: Dict) -> str:
        """Call the LLM API and return the raw text response."""
        ...

    @abstractmethod
    def _build_state_update(self, result: Dict, state: Dict) -> Dict:
        """
        Turn the parsed LLM result into the state keys this agent owns.
        Only return keys that this agent is responsible for.
        """
        ...

    @abstractmethod
    def _fallback(self, state: Dict) -> Dict:
        """
        Return a safe default state update when all API calls fail.
        Should lean toward caution (approve=False or neutral verdict).
        """
        ...

    # ── Shared helpers ────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        """
        Robustly parse JSON from the LLM response,
        stripping markdown code fences if present.
        """
        if not raw:
            return {}
        text = raw.strip()
        # Strip markdown code block if present
        if "```" in text:
            for part in text.split("```"):
                if "{" in part:
                    text = part.replace("json", "").strip()
                    break
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: try to extract the first {...} block
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except Exception:
                    pass
        return {}

    @staticmethod
    def _direction_from_signal(state: Dict) -> str:
        sig = state.get('signal') or {}
        return sig.get('type', 'unknown')

    @staticmethod
    def _format_signal_context(state: Dict) -> str:
        """Build a compact human-readable summary for LLM prompts."""
        sig  = state.get('signal') or {}
        ctx  = state.get('context') or {}
        hour = state.get('hour', 0)

        if   hour in [8, 9]:    phase = "London Open"
        elif hour in [13, 14]:  phase = "NY Kill Zone"
        elif hour in [15, 16]:  phase = "NY AM Silver Bullet"
        elif hour in [19, 20]:  phase = "NY PM Silver Bullet"
        else:                   phase = "Regular Hours"

        return (
            f"Symbol: {state.get('symbol', 'USTECm')} | "
            f"Direction: {sig.get('type', 'unknown').upper()} | "
            f"Price: {state.get('current_price', 0):.2f} | "
            f"Session: {state.get('session', 'new_york')} ({phase}) | "
            f"Quality: {state.get('quality_score', 0):.1f} | "
            f"ADX: {ctx.get('adx', 0):.1f} | "
            f"Regime: {ctx.get('regime', 'unknown')} | "
            f"Vol: {ctx.get('volatility', 'normal')} | "
            f"HTF Bias: {state.get('htf_bias', 'neutral')} | "
            f"Structure: {state.get('structure', 'none')} | "
            f"SL: {state.get('sl_price', 0):.2f} | "
            f"TP: {state.get('tp_price', 0):.2f} | "
            f"Balance: ${state.get('balance', 0):.2f}"
        )
