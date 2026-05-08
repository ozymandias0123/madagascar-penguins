"""
core/circuit_breaker.py — Emergency stop / cooldown mechanism.
Ported from ozy.py / CircuitBreaker unchanged.
"""

import logging

import pandas as pd

from config import Config


class CircuitBreaker:

    def __init__(self, initial_balance: float):
        self.initial_balance      = initial_balance
        self.daily_start_balance  = initial_balance
        self.consecutive_losses   = 0
        self.is_triggered         = False
        self.trigger_time         = None
        self.trigger_reason       = None
        self.last_daily_reset     = pd.Timestamp.now(tz='UTC').date()
        logging.info(f"[CIRCUIT_BREAKER] 🛡️ Initialised with ${initial_balance:.2f}")

    def reset_daily(self, current_balance: float):
        today = pd.Timestamp.now(tz='UTC').date()
        if today > self.last_daily_reset:
            self.daily_start_balance = current_balance
            self.last_daily_reset    = today

    def check_and_trigger(self, stats_tracker) -> bool:
        if Config.BACKTEST_MODE or not Config.CIRCUIT_BREAKER_ENABLED:
            return True
        if self.is_triggered:
            return self._check_cooldown()

        stats           = stats_tracker.get_stats()
        current_balance = stats['current_balance']
        self.reset_daily(current_balance)

        if stats['max_drawdown'] >= Config.LIVE_MAX_DRAWDOWN:
            self._trigger(f"MAX_DRAWDOWN ({stats['max_drawdown']:.1%})")
            return False
        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance
        if daily_loss >= Config.LIVE_DAILY_LOSS_LIMIT:
            self._trigger(f"DAILY_LOSS ({daily_loss:.1%})")
            return False
        if current_balance < self.initial_balance * Config.LIVE_MIN_BALANCE_PERCENT:
            self._trigger(f"LOW_BALANCE (${current_balance:.2f})")
            return False
        if self.consecutive_losses >= Config.LIVE_CONSECUTIVE_LOSSES:
            self._trigger(f"CONSECUTIVE_LOSSES ({self.consecutive_losses})")
            return False
        return True

    def record_trade_result(self, is_win: bool):
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            logging.warning(f"[CIRCUIT_BREAKER] ❌ Consecutive losses: {self.consecutive_losses}")

    def _trigger(self, reason: str):
        self.is_triggered   = True
        self.trigger_time   = pd.Timestamp.now(tz='UTC')
        self.trigger_reason = reason
        logging.critical(f"[CIRCUIT_BREAKER] 🚨 EMERGENCY STOP: {reason}")

    def _check_cooldown(self) -> bool:
        if self.trigger_time is None:
            return True
        elapsed = (pd.Timestamp.now(tz='UTC') - self.trigger_time).total_seconds() / 3600
        if elapsed >= Config.CIRCUIT_BREAKER_COOLDOWN_HOURS:
            self.is_triggered   = False
            self.trigger_time   = None
            self.trigger_reason = None
            self.consecutive_losses = 0
            logging.info("[CIRCUIT_BREAKER] ✅ Cooldown ended — re-armed")
            return True
        return False

    def force_reset(self):
        self.is_triggered       = False
        self.trigger_time       = None
        self.trigger_reason     = None
        self.consecutive_losses = 0
        logging.info("[CIRCUIT_BREAKER] 🔄 Force reset")
