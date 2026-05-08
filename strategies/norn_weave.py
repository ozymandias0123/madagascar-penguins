"""
strategies/norn_weave.py
NORN WEAVE URUZ — EMA + Dow Structure + ADX
Focus-based EMA, pivot swing highs/lows, ADX trend confirmation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class NornWeave(BaseStrategy):

    name        = "NornWeave"
    description = "Focus EMA rising/falling + Dow swing structure + ADX > 20"
    version     = "1.0"

    FOCUS    = 13
    ADX_MIN  = 20

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        ema_len = self.FOCUS * 5   # 65
        if len(df) < ema_len + 10:
            return []

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()
        adx   = df["adx"] if "adx" in df.columns else pd.Series(25.0, index=df.index)

        ema = close.ewm(span=ema_len, adjust=False).mean()

        # rising / falling: EMA increasing for last 2 bars
        ema_rising  = (ema > ema.shift(1)) & (ema.shift(1) > ema.shift(2))
        ema_falling = (ema < ema.shift(1)) & (ema.shift(1) < ema.shift(2))

        # Dow structure: pivot highs and lows
        sw = self.FOCUS
        roll_max = high.rolling(2 * sw + 1, center=True).max()
        roll_min = low.rolling(2 * sw + 1, center=True).min()
        pivot_h  = high.where(high == roll_max)
        pivot_l  = low.where(low  == roll_min)

        last_h = pivot_h.dropna().iloc[-1] if not pivot_h.dropna().empty else np.nan
        last_l = pivot_l.dropna().iloc[-1] if not pivot_l.dropna().empty else np.nan

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        adx_v = float(adx.iloc[i])

        trend_up   = entry > last_l if not np.isnan(last_l) else False
        trend_down = entry < last_h if not np.isnan(last_h) else False

        signals = []

        if ema_rising.iloc[i] and trend_up and adx_v > self.ADX_MIN:
            sl = entry * 0.97           # 3% hard stop (original strategy)
            tp = entry + atr_v * 4.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.0,
                "zone":        {"high": entry + atr_v * 0.3, "low": last_l if not np.isnan(last_l) else entry - atr_v},
                "pattern_key": "norn_dow_long",
                "strategy":    self.name,
                "notes":       f"EMA{ema_len} rising, Dow uptrend, ADX={adx_v:.1f}",
            })

        if ema_falling.iloc[i] and trend_down and adx_v > self.ADX_MIN:
            sl = entry * 1.03
            tp = entry - atr_v * 4.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.0,
                "zone":        {"high": last_h if not np.isnan(last_h) else entry + atr_v, "low": entry - atr_v * 0.3},
                "pattern_key": "norn_dow_short",
                "strategy":    self.name,
                "notes":       f"EMA{ema_len} falling, Dow downtrend, ADX={adx_v:.1f}",
            })

        return signals
