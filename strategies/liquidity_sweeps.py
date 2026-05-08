"""
strategies/liquidity_sweeps.py
Liquidity Sweeps + Sessions
Stop-hunt detection at pivot highs/lows — entry on sweep reversal
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class LiquiditySweeps(BaseStrategy):

    name        = "LiquiditySweeps"
    description = "Pivot sweep detection — price wicks through level then reverses"
    version     = "1.0"

    SWING = 5
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

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()

        sw = self.SWING
        pivot_h = high.rolling(2 * sw + 1, center=True).max()
        pivot_h = high.where(high == pivot_h).shift(sw)   # confirmed pivot

        pivot_l = low.rolling(2 * sw + 1, center=True).min()
        pivot_l = low.where(low == pivot_l).shift(sw)

        # Sweep high: price crossed ABOVE previous pivot high then closed BELOW it
        # (stop hunt on sell-side — bullish reversal)
        prev_ph = pivot_h.shift(1).ffill()
        sweep_low = (high > prev_ph) & (close < prev_ph)

        # Sweep low: price crossed BELOW previous pivot low then closed ABOVE it
        # (stop hunt on buy-side — bearish reversal... wait, sweep below pivot LOW = liquidity grab → expect up)
        prev_pl = pivot_l.shift(1).ffill()
        sweep_high = (low < prev_pl) & (close > prev_pl)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        signals = []

        # Sweep below pivot low → price grabbed sell-stops → expect up
        if sweep_high.iloc[i]:
            sl = float(last["low"]) - atr_v * 0.3
            tp = entry + atr_v * 2.5
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": entry + atr_v * 0.2, "low": float(last["low"])},
                "pattern_key": "liquidity_sweep_long",
                "strategy":    self.name,
                "notes":       f"Stop-hunt below pivot low, reversal buy, session={session}",
            })

        # Sweep above pivot high → price grabbed buy-stops → expect down
        if sweep_low.iloc[i]:
            sl = float(last["high"]) + atr_v * 0.3
            tp = entry - atr_v * 2.5
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": float(last["high"]), "low": entry - atr_v * 0.2},
                "pattern_key": "liquidity_sweep_short",
                "strategy":    self.name,
                "notes":       f"Stop-hunt above pivot high, reversal sell, session={session}",
            })

        return signals
