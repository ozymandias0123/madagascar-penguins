"""
strategies/multi_factor_confluence.py
Multi-Factor Confluence Strategy
EMA trend + price pressure score → crossover entry
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class MultiFactorConfluence(BaseStrategy):

    name        = "MultiFactorConfluence"
    description = "EMA trend + pressure confluence score → crossover entry"
    version     = "1.0"

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 55:
            return []

        close = df["close"]
        atr   = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()

        ema_fast = close.ewm(span=21, adjust=False).mean()
        ema_slow = close.ewm(span=50, adjust=False).mean()
        sma20    = close.rolling(20).mean()
        atr_val  = atr.rolling(14).mean() if "atr" not in df.columns else atr

        trend_bull = ema_fast > ema_slow

        pressure   = (close - sma20) / atr_val.replace(0, np.nan)
        score_long = trend_bull.astype(int) * 2 - 1 + (pressure > 0).astype(int) * 2 - 1

        # crossover: close crosses above ema_fast
        cross_up   = (close.shift(1) <= ema_fast.shift(1)) & (close > ema_fast)
        cross_down = (close.shift(1) >= ema_fast.shift(1)) & (close < ema_fast)

        i    = -2   # last closed candle
        last = df.iloc[i]
        entry  = float(last["close"])
        atr_v  = float(atr_val.iloc[i]) if not np.isnan(atr_val.iloc[i]) else 10.0
        sl_atr = 1.5
        tp_atr = 3.0

        signals = []

        if score_long.iloc[i] >= 2 and cross_up.iloc[i]:
            sl = entry - atr_v * sl_atr
            tp = entry + atr_v * tp_atr
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.0,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "mfc_long",
                "strategy":    self.name,
                "notes":       f"Confluence score={score_long.iloc[i]:.0f}, EMA cross up",
            })

        score_short = -score_long
        if score_short.iloc[i] >= 2 and cross_down.iloc[i]:
            sl = entry + atr_v * sl_atr
            tp = entry - atr_v * tp_atr
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.0,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "mfc_short",
                "strategy":    self.name,
                "notes":       f"Confluence score={score_short.iloc[i]:.0f}, EMA cross down",
            })

        return signals
