"""
core/exchange/base_exchange.py — Unified exchange interface.

Every exchange adapter (MT5, Binance, Bybit, …) inherits from BaseExchange.
The engine only calls methods defined here — making exchange swapping seamless.

Timeframe convention used throughout: CCXT-style strings
  '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


# ── Data transfer objects ─────────────────────────────────────────────────────

@dataclass
class OrderResult:
    success:     bool
    ticket:      int    = 0
    symbol:      str    = ""
    order_type:  str    = ""          # 'buy' | 'sell'
    lot:         float  = 0.0
    entry_price: float  = 0.0
    sl_price:    float  = 0.0
    tp_price:    float  = 0.0
    error:       str    = ""
    raw:         Dict   = field(default_factory=dict)


@dataclass
class PositionInfo:
    ticket:    int
    symbol:    str
    order_type: str    # 'buy' | 'sell'
    volume:    float
    price:     float   # entry price
    current:   float   # current price
    profit:    float
    sl:        float
    tp:        float
    comment:   str = ""


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseExchange(ABC):
    """
    Unified interface for all exchanges.

    Design rules:
    • connect() must be called before any other method.
    • All methods return safe defaults on error (never raise).
    • Timeframes use CCXT convention ('15m', '1h', '4h', …).
    • Lot sizes are always in base-asset units for crypto,
      and in micro-lots for MT5.
    """

    name:             str  = "BaseExchange"
    supports_futures: bool = False
    supports_spot:    bool = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and open connection. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection and release resources."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    # ── Account ───────────────────────────────────────────────────────────────

    @abstractmethod
    def get_balance(self) -> float:
        """Free / available balance in USD (or account currency)."""
        ...

    @abstractmethod
    def get_account_info(self) -> Dict:
        """Return dict with at least: balance, equity, margin_free, leverage."""
        ...

    # ── Market data ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Current last-traded price."""
        ...

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """
        OHLCV DataFrame — columns: time, open, high, low, close, volume.
        Rows in ascending order (oldest first).
        """
        ...

    @abstractmethod
    def get_spread(self, symbol: str) -> float:
        """Current bid-ask spread in price units (0 if not available)."""
        ...

    # ── Trading ───────────────────────────────────────────────────────────────

    @abstractmethod
    def place_order(
        self,
        symbol:     str,
        order_type: str,     # 'buy' | 'sell'
        lot:        float,
        sl_price:   float,
        tp_price:   float,
        comment:    str = "",
    ) -> OrderResult: ...

    @abstractmethod
    def close_position(self, ticket: int, symbol: str = "") -> bool: ...

    @abstractmethod
    def get_open_positions(self) -> List[PositionInfo]: ...

    # ── Risk helpers ──────────────────────────────────────────────────────────

    @abstractmethod
    def calculate_lot(
        self,
        balance:     float,
        risk_pct:    float,   # 0.005 = 0.5 %
        sl_distance: float,   # price distance from entry to SL
        symbol:      str,
    ) -> float:
        """Return lot/contract size respecting exchange minimum & precision."""
        ...

    # ── Shared utilities ──────────────────────────────────────────────────────

    @staticmethod
    def tf_to_seconds(timeframe: str) -> int:
        """Convert CCXT timeframe string to seconds."""
        _MAP = {
            '1m': 60,   '3m': 180,   '5m': 300,   '15m': 900,
            '30m': 1800, '1h': 3600,  '2h': 7200,  '4h': 14400,
            '6h': 21600, '8h': 28800, '12h': 43200, '1d': 86400,
        }
        return _MAP.get(timeframe, 900)

    @staticmethod
    def mt5_tf_to_ccxt(mt5_tf) -> str:
        """Convert MetaTrader5 timeframe constant → CCXT string."""
        try:
            import MetaTrader5 as mt5
            _MAP = {
                mt5.TIMEFRAME_M1:  '1m',
                mt5.TIMEFRAME_M5:  '5m',
                mt5.TIMEFRAME_M15: '15m',
                mt5.TIMEFRAME_M30: '30m',
                mt5.TIMEFRAME_H1:  '1h',
                mt5.TIMEFRAME_H4:  '4h',
                mt5.TIMEFRAME_D1:  '1d',
            }
            return _MAP.get(mt5_tf, '15m')
        except ImportError:
            return '15m'

    @staticmethod
    def ccxt_tf_to_mt5(ccxt_tf: str):
        """Convert CCXT timeframe string → MetaTrader5 timeframe constant."""
        try:
            import MetaTrader5 as mt5
            _MAP = {
                '1m':  mt5.TIMEFRAME_M1,
                '5m':  mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,
                '30m': mt5.TIMEFRAME_M30,
                '1h':  mt5.TIMEFRAME_H1,
                '4h':  mt5.TIMEFRAME_H4,
                '1d':  mt5.TIMEFRAME_D1,
            }
            return _MAP.get(ccxt_tf, mt5.TIMEFRAME_M15)
        except ImportError:
            return None
