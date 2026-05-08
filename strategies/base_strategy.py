"""
strategies/base_strategy.py

Every strategy must inherit from BaseStrategy and implement:
  - generate_signals(df, context, session, htf_bias) -> list[dict]

Signal dict format (returned by generate_signals):
{
    "type":         "buy" | "sell",
    "entry_price":  float,
    "sl_price":     float,
    "tp_price":     float,
    "quality":      float,          # 1-10
    "zone":         dict,           # {"high": float, "low": float}
    "pattern_key":  str,            # e.g. "bos_fvg_buy"
    "strategy":     str,            # strategy name
    "notes":        str,            # human-readable reason
}
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base for all trading strategies.
    Subclass this, set `name`, implement `generate_signals`.
    """

    # ---- override in each strategy ----
    name:        str = "BaseStrategy"
    description: str = ""
    version:     str = "1.0"

    # ---- shared helpers ---------------

    @abstractmethod
    def generate_signals(
        self,
        df:       pd.DataFrame,
        context:  Dict[str, Any],
        session:  str,
        htf_bias: str,
    ) -> List[Dict]:
        """
        Analyse the dataframe and return a list of signal dicts.
        Return [] if no trade setup is found.

        Parameters
        ----------
        df       : OHLCV dataframe with indicators already computed
                   (rsi, ema_fast, ema_slow, atr, adx, etc.)
        context  : {'adx', 'atr_ratio', 'volatility', 'regime', ...}
        session  : 'london' | 'new_york' | 'asian' | 'off'
        htf_bias : 'bullish' | 'bearish' | 'neutral'
        """
        ...

    # ---- optional hooks (override if needed) ----

    def on_trade_closed(self, result: Dict) -> None:
        """Called after a trade closes. Use for strategy-level learning."""
        pass

    def validate_signal(self, signal: Dict, balance: float) -> bool:
        """
        Extra sanity checks before signal reaches the orchestrator.
        Default: always passes.
        """
        return True

    # ---- shared utilities ----

    @staticmethod
    def _candle_body(row) -> float:
        return abs(row["close"] - row["open"])

    @staticmethod
    def _candle_range(row) -> float:
        return row["high"] - row["low"]

    @staticmethod
    def _is_bullish(row) -> bool:
        return row["close"] > row["open"]

    @staticmethod
    def _is_bearish(row) -> bool:
        return row["close"] < row["open"]

    @staticmethod
    def _rr(entry: float, sl: float, tp: float) -> float:
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        return round(reward / risk, 2) if risk > 0 else 0.0

    def __repr__(self):
        return f"<Strategy: {self.name} v{self.version}>"
