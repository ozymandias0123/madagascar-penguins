"""
strategies/gann_phases.py
Gann Phases Strategy
EMA8/21 + pivot swing + angle-based phase filter
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class GannPhases(BaseStrategy):

    name        = "GannPhases"
    description = "Gann angle phase (ACCUM/MODER/EXPAN) + EMA + swing pivot"
    version     = "1.0"

    SWING_LEN = 5

    def _pivot_high(self, high: pd.Series, left: int, right: int) -> pd.Series:
        n   = left + right + 1
        rolling_max = high.rolling(n, center=True).max()
        return high.where(high == rolling_max)

    def _pivot_low(self, low: pd.Series, left: int, right: int) -> pd.Series:
        n   = left + right + 1
        rolling_min = low.rolling(n, center=True).min()
        return low.where(low == rolling_min)

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 30:
            return []

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()

        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()

        # angle proxy (10-bar momentum / atr)
        angle = (close - close.shift(10)) / atr_s.replace(0, np.nan) * 100

        def phase(a):
            if a > 2:   return "EXPAN"
            if a > 0.5: return "MODER"
            return "ACCUM"

        # pivot lows/highs
        pl = self._pivot_low(low,   self.SWING_LEN, self.SWING_LEN)
        ph = self._pivot_high(high, self.SWING_LEN, self.SWING_LEN)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        ang   = float(angle.iloc[i])
        ph_   = phase(ang)

        # last pivot values
        last_pl = pl.dropna().iloc[-1] if not pl.dropna().empty else entry - atr_v * 2
        last_ph = ph.dropna().iloc[-1] if not ph.dropna().empty else entry + atr_v * 2

        bull_swing = entry > last_pl
        bear_swing = entry < last_ph

        candle_bull = last["close"] > last["open"]
        candle_bear = last["close"] < last["open"]

        signals = []

        if (bull_swing and entry > float(ema8.iloc[i])
                and candle_bull and ph_ != "ACCUM"):
            sl = entry - atr_v * 1.5
            tp = entry + atr_v * 3.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.5,
                "zone":        {"high": entry + atr_v * 0.4, "low": last_pl},
                "pattern_key": f"gann_{ph_.lower()}_long",
                "strategy":    self.name,
                "notes":       f"Phase={ph_}, bull swing above EMA8",
            })

        if (bear_swing and entry < float(ema8.iloc[i])
                and candle_bear and ph_ != "ACCUM"):
            sl = entry + atr_v * 1.5
            tp = entry - atr_v * 3.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.5,
                "zone":        {"high": last_ph, "low": entry - atr_v * 0.4},
                "pattern_key": f"gann_{ph_.lower()}_short",
                "strategy":    self.name,
                "notes":       f"Phase={ph_}, bear swing below EMA8",
            })

        return signals
