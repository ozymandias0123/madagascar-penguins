"""
agents/trade_engine.py — TradeEngine (Execution)
Model  : Rule-based (no LLM)
Role   : Executes trades the moment Private gives the green light.
         No second thoughts — just action.
Node   : execution_node  ->  updates state['execution_result']
"""

import logging
from typing import Dict, Optional

from config import Config


class TradeEngine:
    """
    TradeEngine — pure rule-based execution.
    Wraps the engine's _execute_trade() with position-size adjustment
    from Kowalski's risk assessment.
    """

    name = "TradeEngine"

    def __init__(self, engine=None):
        self._engine = engine

    # -- Public interface ----------------------------------------

    def run(self, state: Dict) -> Dict:
        """Execute the trade. Private already said go."""
        if not state.get('approved', False):
            logging.info("[TradeEngine] Not approved — standing down")
            return {
                'execution_result': {
                    'executed': False,
                    'reason':   'not_approved',
                    'action':   'skip',
                }
            }

        final_action = state.get('final_action', 'skip')
        if final_action not in ('buy', 'sell'):
            logging.info(f"[TradeEngine] Action='{final_action}' — nothing to fire")
            return {
                'execution_result': {
                    'executed': False,
                    'reason':   'invalid_action',
                    'action':   final_action,
                }
            }

        signal        = state.get('signal') or {}
        zone          = signal.get('zone', {})
        entry_price   = state.get('entry_price', signal.get('entry_price', 0.0))
        atr           = state.get('atr', 10.0)
        session       = state.get('session', 'new_york')
        structure     = state.get('structure', 'no_structure')
        quality_score = state.get('quality_score', 1.0)

        ra       = state.get('risk_assessment') or {}
        size_adj = ra.get('position_size_adj', 1.0)

        logging.info(
            f"[TradeEngine] FIRING: {final_action.upper()} | "
            f"Entry={entry_price:.2f} | "
            f"Quality={quality_score:.1f} | "
            f"SizeAdj={size_adj:.2f} | "
            f"Session={session}"
        )

        if self._engine is None:
            logging.warning("[TradeEngine] No engine attached — DRY RUN")
            return {
                'execution_result': {
                    'executed':    False,
                    'reason':      'dry_run',
                    'action':      final_action,
                    'entry_price': entry_price,
                    'session':     session,
                    'quality':     quality_score,
                    'size_adj':    size_adj,
                }
            }

        # -- Fire! -----------------------------------------------
        _orig_risk = Config.RISK_PERCENT
        try:
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

            Config.RISK_PERCENT = _orig_risk
            logging.info(f"[TradeEngine] {final_action.upper()} fired successfully")

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
            Config.RISK_PERCENT = _orig_risk
            logging.error(f"[TradeEngine] Misfire! {exc}")
            return {
                'execution_result': {
                    'executed': False,
                    'reason':   f'exception: {exc}',
                    'action':   final_action,
                }
            }

    # -- Engine setter -------------------------------------------

    def set_engine(self, engine) -> None:
        self._engine = engine
        logging.info("[TradeEngine] Engine loaded — ready to fire")
