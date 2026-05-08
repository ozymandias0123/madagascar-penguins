"""
agents/execution_agent.py — RicoExecution
Pure rule-based execution agent — no LLM, no API calls.
Reads the final approved signal from state and delegates to the engine's
_execute_trade() method.  Returns execution metadata in state.

Node: execution_node  →  updates state['execution_result']
"""

import logging
from typing import Dict, Optional

from config import Config


class RicoExecution:
    """
    Rule-based execution agent.

    Because it needs to call engine._execute_trade(), it receives
    the engine reference at construction time.  If no engine is passed
    (e.g. during unit tests) it logs a dry-run result instead.
    """

    name = "RicoExecution"

    def __init__(self, engine=None):
        """
        Parameters
        ----------
        engine : PersistentTradingEngine | None
            Live engine instance.  Pass None for dry-run / test mode.
        """
        self._engine = engine

    # ── Public interface (mirrors BaseAgent.run signature) ────

    def run(self, state: Dict) -> Dict:
        """Execute the approved trade and return a partial state update."""
        if not state.get('approved', False):
            logging.info("[RicoExecution] ⏭️  Trade not approved — skipping execution")
            return {
                'execution_result': {
                    'executed': False,
                    'reason':   'not_approved',
                    'action':   'skip'
                }
            }

        final_action = state.get('final_action', 'skip')
        if final_action not in ('buy', 'sell'):
            logging.info(f"[RicoExecution] ⏭️  final_action={final_action} — skipping")
            return {
                'execution_result': {
                    'executed': False,
                    'reason':   'invalid_action',
                    'action':   final_action
                }
            }

        signal        = state.get('signal') or {}
        zone          = signal.get('zone', {})
        entry_price   = state.get('entry_price', signal.get('entry_price', 0.0))
        atr           = state.get('atr', 10.0)
        session       = state.get('session', 'new_york')
        structure     = state.get('structure', 'no_structure')
        quality_score = state.get('quality_score', 1.0)

        # Risk Manager may have provided a position-size adjustment
        ra            = state.get('risk_assessment') or {}
        size_adj      = ra.get('position_size_adj', 1.0)

        logging.info(
            f"[RicoExecution] 🚀 Executing {final_action.upper()} "
            f"| Entry={entry_price:.2f} | Quality={quality_score:.1f} "
            f"| SizeAdj={size_adj:.2f} | Session={session}"
        )

        if self._engine is None:
            # Dry-run mode (used in tests / backtest without engine ref)
            logging.warning("[RicoExecution] ⚠️  No engine attached — DRY RUN")
            return {
                'execution_result': {
                    'executed':     False,
                    'reason':       'dry_run',
                    'action':       final_action,
                    'entry_price':  entry_price,
                    'session':      session,
                    'quality':      quality_score,
                }
            }

        try:
            # Temporarily apply position-size multiplier via Config
            _orig_risk = Config.RISK_PERCENT
            Config.RISK_PERCENT = _orig_risk * size_adj

            self._engine._execute_trade(
                order_type    = final_action,
                zone          = zone,
                entry_price   = entry_price,
                atr           = atr,
                session       = session,
                structure     = structure,
                quality_score = quality_score,
                signal        = signal,
            )

            Config.RISK_PERCENT = _orig_risk   # restore

            logging.info(
                f"[RicoExecution] ✅ _execute_trade() completed "
                f"for {final_action.upper()}"
            )
            return {
                'execution_result': {
                    'executed':    True,
                    'action':      final_action,
                    'entry_price': entry_price,
                    'session':     session,
                    'quality':     quality_score,
                    'size_adj':    size_adj,
                }
            }

        except Exception as exc:
            logging.error(f"[RicoExecution] ❌ Execution failed: {exc}")
            Config.RISK_PERCENT = _orig_risk   # always restore
            return {
                'execution_result': {
                    'executed': False,
                    'reason':   f'exception: {exc}',
                    'action':   final_action,
                }
            }

    # ── Engine setter (allows late binding) ───────────────────

    def set_engine(self, engine) -> None:
        self._engine = engine
        logging.info("[RicoExecution] Engine reference set")
