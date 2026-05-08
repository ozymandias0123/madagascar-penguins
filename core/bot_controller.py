"""
core/bot_controller.py — Thread-safe bot control bridge.

The trading engine runs in the main thread.
Private (Telegram bot) runs in a daemon thread.
BotController is the shared singleton they both talk through.

Engine calls:
    ctrl.wait_if_paused()      — blocks until resumed
    ctrl.is_stopped()          — check if engine should exit
    ctrl.get_mode()            — 'demo'|'live'|'signals'|'analysis'
    ctrl.get_active_exchange() — current exchange name

Private (Telegram) calls:
    ctrl.pause()   ctrl.resume()
    ctrl.stop()
    ctrl.set_mode('demo')
    ctrl.set_active_exchange('bybit')
    ctrl.set_symbol('BTC/USDT:USDT')
    ctrl.set_risk(0.01)
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class BotController:
    """Singleton controller shared between engine and Telegram bot threads."""

    _instance: Optional["BotController"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        from config import Config

        self._pause_event   = threading.Event()
        self._stop_event    = threading.Event()
        # Start PAUSED — engine waits for /demo /live /signals from Telegram

        self._mode:     str   = Config.MODE          # 'demo'|'live'|'signals'|'analysis'
        self._exchange: str   = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5')
        self._symbol:   str   = Config.SYMBOLS[0] if Config.SYMBOLS else 'USTECm'
        self._risk:     float = Config.RISK_PERCENT

        self._engine_ref = None   # set by engine after start
        self._state_lock = threading.Lock()

    # ── Engine hooks ──────────────────────────────────────────────────────────

    def set_engine(self, engine) -> None:
        self._engine_ref = engine

    def wait_if_paused(self, timeout: float = 60.0) -> None:
        """Block the engine loop while paused. Returns when resumed or timeout."""
        self._pause_event.wait(timeout=timeout)

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    # ── Telegram control commands ─────────────────────────────────────────────

    def pause(self) -> str:
        if self.is_paused():
            return "⏸ Already paused."
        self._pause_event.clear()
        logger.info("[BotController] ⏸ Trading PAUSED by Telegram")
        return "⏸ Trading paused. Send /resume to continue."

    def resume(self) -> str:
        if self._stop_event.is_set():
            return "❌ Bot is stopped. Restart the process."
        if not self.is_paused():
            return "▶️ Already running."
        self._pause_event.set()
        logger.info("[BotController] ▶️ Trading RESUMED by Telegram")
        return "▶️ Trading resumed."

    def stop(self) -> str:
        self._stop_event.set()
        self._pause_event.set()   # unblock wait_if_paused so engine can exit cleanly
        logger.info("[BotController] 🛑 Bot STOP requested via Telegram")
        return "🛑 Stop signal sent. The bot will finish the current cycle and exit."

    def set_mode(self, mode: str) -> str:
        mode = mode.strip().lower()
        valid = {'demo', 'live', 'signals', 'analysis', 'backtest'}
        if mode not in valid:
            return f"❌ Invalid mode. Choose: {', '.join(sorted(valid))}"
        with self._state_lock:
            old = self._mode
            self._mode = mode
        # Update Config too so agents read the new value
        try:
            from config import Config
            Config.MODE          = mode
            Config.PAPER_TRADING = mode == 'demo'
            Config.LIVE_MODE     = mode == 'live'
            Config.BACKTEST_MODE = mode == 'backtest'
        except Exception:
            pass
        logger.info(f"[BotController] Mode changed: {old} → {mode}")
        return f"⚙️ Mode switched to <b>{mode.upper()}</b>."

    def set_active_exchange(self, name: str) -> str:
        from core.exchange import EXCHANGE_REGISTRY
        name = name.strip().lower()
        if name not in EXCHANGE_REGISTRY:
            available = ', '.join(sorted(EXCHANGE_REGISTRY))
            return f"❌ Unknown exchange. Available:\n<code>{available}</code>"
        with self._state_lock:
            old = self._exchange
            self._exchange = name
        try:
            from config import Config
            Config.ACTIVE_EXCHANGE = name
        except Exception:
            pass
        logger.info(f"[BotController] Exchange changed: {old} → {name}")
        return f"🔀 Exchange switched to <b>{name.upper()}</b>.\n⚠️ Restart for full effect."

    def set_symbol(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        with self._state_lock:
            old = self._symbol
            self._symbol = symbol
        try:
            from config import Config
            Config.SYMBOLS = [symbol]
        except Exception:
            pass
        logger.info(f"[BotController] Symbol changed: {old} → {symbol}")
        return f"📡 Symbol changed to <b>{symbol}</b>."

    def set_risk(self, pct: float) -> str:
        if not (0.001 <= pct <= 0.05):
            return "❌ Risk must be between 0.1% and 5% (e.g. 0.01 = 1%)."
        with self._state_lock:
            old = self._risk
            self._risk = pct
        try:
            from config import Config
            Config.RISK_PERCENT = pct
        except Exception:
            pass
        logger.info(f"[BotController] Risk changed: {old:.3f} → {pct:.3f}")
        return f"⚡ Risk per trade: <b>{pct*100:.2f}%</b>."

    # ── Getters ───────────────────────────────────────────────────────────────

    def get_mode(self) -> str:
        with self._state_lock:
            return self._mode

    def get_active_exchange(self) -> str:
        with self._state_lock:
            return self._exchange

    def get_symbol(self) -> str:
        with self._state_lock:
            return self._symbol

    def get_risk(self) -> float:
        with self._state_lock:
            return self._risk

    def get_status_text(self) -> str:
        mode     = self.get_mode()
        exchange = self.get_active_exchange()
        symbol   = self.get_symbol()
        risk     = self.get_risk()
        state    = "⏸ PAUSED" if self.is_paused() else ("🛑 STOPPED" if self.is_stopped() else "🟢 RUNNING")
        mode_icons = {'live': '🔴', 'demo': '🟡', 'signals': '📡', 'analysis': '🔬', 'backtest': '🔵'}
        icon = mode_icons.get(mode, '⚙️')

        return (
            f"🤖 <b>BOT STATUS</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"State:    <code>{state}</code>\n"
            f"Mode:     <code>{icon} {mode.upper()}</code>\n"
            f"Exchange: <code>{exchange.upper()}</code>\n"
            f"Symbol:   <code>{symbol}</code>\n"
            f"Risk:     <code>{risk*100:.2f}%</code>"
        )


def get_controller() -> BotController:
    """Return the singleton BotController instance."""
    return BotController()
