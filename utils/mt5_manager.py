"""
utils/mt5_manager.py — MetaTrader 5 connection & symbol validation.
Extracted from ozy.py unchanged.
"""

import logging
import time
import pandas as pd
import MetaTrader5 as mt5
from functools import wraps
from typing import Optional

from config import Config


# ── Decorators ────────────────────────────────────────────────

def mt5_error_handler(func):
    """Retry decorator for MT5 API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(Config.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logging.error(
                    f"[MT5_ERROR] {func.__name__} failed: {exc} "
                    f"(attempt {attempt + 1}) | MT5: {mt5.last_error()}"
                )
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY * (2 ** attempt))
                else:
                    logging.critical("[MAX_RETRIES] Max retries reached — pausing 1 h")
                    time.sleep(3600)
                    raise
    return wrapper


def timed_function(func):
    """Logs how long a function takes (DEBUG level)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.time()
        result = func(*args, **kwargs)
        logging.debug(f"[TIMER] {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper


# ── MT5Manager ────────────────────────────────────────────────

class MT5Manager:
    """Handles MT5 initialisation, login, and symbol selection."""

    @staticmethod
    @mt5_error_handler
    def initialize() -> str:
        Config.validate()
        if not mt5.initialize():
            raise ConnectionError(f"[MT5_INIT_ERROR] {mt5.last_error()}")

        account_info = mt5.account_info()
        if account_info is not None:
            logging.info(f"[MT5_CONNECTED] Already logged in — Account #{account_info.login}")
            return MT5Manager.get_valid_symbol()

        # Explicit login
        try:
            if not mt5.login(
                login=int(Config.LOGIN),
                password=Config.PASSWORD,
                server=Config.SERVER
            ):
                raise ConnectionError(f"[MT5_LOGIN_ERROR] {mt5.last_error()}")
        except Exception as exc:
            raise ConnectionError(f"MT5 login failed: {exc}")

        return MT5Manager.get_valid_symbol()

    @staticmethod
    def _validate_symbol(symbol: str) -> bool:
        info = mt5.symbol_info(symbol)
        if info is None or not info.visible:
            return mt5.symbol_select(symbol, True)
        return True

    @staticmethod
    def get_valid_symbol() -> str:
        now = pd.Timestamp.now(tz='UTC')
        for symbol in Config.SYMBOLS:
            if not MT5Manager._validate_symbol(symbol):
                continue
            rates = mt5.copy_rates_range(
                symbol, Config.TIMEFRAME,
                now - pd.Timedelta(days=30), now
            )
            if rates is not None and len(rates) >= Config.MIN_CANDLES:
                logging.info(f"[SYMBOL_SELECTED] {symbol} — {len(rates)} candles available")
                return symbol
        raise ValueError(f"[NO_VALID_SYMBOLS] {Config.SYMBOLS} | MT5: {mt5.last_error()}")

    @staticmethod
    def ensure_connected() -> bool:
        """Re-initialise MT5 if the connection dropped. Returns True if connected."""
        if mt5.account_info() is not None:
            return True
        try:
            MT5Manager.initialize()
            return True
        except Exception as exc:
            logging.error(f"[MT5_RECONNECT] {exc}")
            return False
