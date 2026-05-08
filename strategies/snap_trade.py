"""
strategies/snap_trade.py
Snap Trade Strategy — Morning Momentum Setup

Strong momentum burst:
  - price above EMA9
  - 3-bar ROC > 0.4% (long) or < -0.4% (short)
  - volume ratio > 1.8x SMA
  - breakout above 5-bar rolling high (long) or below 5-bar rolling low (short)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SnapTrade(BaseStrategy):

    name        = "SnapTrade"
    description = "Morning momentum: EMA9 + ROC + volume surge + rolling breakout"
    version     = "1.0"

    ROC_BARS    = 3
    ROC_MIN     = 0.4     # %
    VOL_RATIO   = 1.8
    BREAK_BARS  = 5
    ACTIVE_SESSIONS = {"london", "new_york"}

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 30:
            return []

        if session not in self.ACTIVE_SESSIONS:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"]   if "atr"    in df.columns else self._calc_atr(df, 14)

        ema9      = close.ewm(span=9, adjust=False).mean()
        roc       = close.pct_change(self.ROC_BARS) * 100
        vol_ratio = volume / volume.rolling(20).mean().replace(0, 1)

        # rolling breakout levels
        rolling_high = high.rolling(self.BREAK_BARS).max().shift(1)
        rolling_low  = low.rolling(self.BREAK_BARS).min().shift(1)

        snap_long = (
            (close > ema9) &
            (roc > self.ROC_MIN) &
            (vol_ratio > self.VOL_RATIO) &
            (close > rolling_high)
        )

        snap_short = (
            (close < ema9) &
            (roc < -self.ROC_MIN) &
            (vol_ratio > self.VOL_RATIO) &
            (close < rolling_low)
        )

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        # Risk box: 3-bar high/low
        box_top    = float(high.rolling(3).max().iloc[i])
        box_bottom = float(low.rolling(3).min().iloc[i])
        roc_v      = float(roc.iloc[i])
        vr_v       = float(vol_ratio.iloc[i])

        signals = []

        if snap_long.iloc[i]:
            sl = box_bottom - atr_v * 0.2
            tp = entry + (entry - sl) * 2.5
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": box_top, "low": box_bottom},
                "pattern_key": "snap_long",
                "strategy":    self.name,
                "notes":       f"Snap long: ROC={roc_v:.2f}%, VolRatio={vr_v:.1f}x, breakout",
            })

        if snap_short.iloc[i]:
            sl = box_top + atr_v * 0.2
            tp = entry - (sl - entry) * 2.5
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": box_top, "low": box_bottom},
                "pattern_key": "snap_short",
                "strategy":    self.name,
                "notes":       f"Snap short: ROC={roc_v:.2f}%, VolRatio={vr_v:.1f}x, breakdown",
            })

        return signals

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
