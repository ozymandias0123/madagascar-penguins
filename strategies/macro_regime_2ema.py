"""
strategies/macro_regime_2ema.py
Macro Regime 2EMA Strategy
EMA50 / EMA200 trend regime + crossover entry
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class MacroRegime2EMA(BaseStrategy):

    name        = "MacroRegime2EMA"
    description = "EMA50/200 macro regime filter + crossover entry"
    version     = "1.0"

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 210:
            return []

        close = df["close"]
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()

        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        trend_bull = (close > ema50) & (ema50 > ema200)
        trend_bear = (close < ema50) & (ema50 < ema200)

        cross_up   = (close.shift(1) <= ema50.shift(1)) & (close > ema50)
        cross_down = (close.shift(1) >= ema50.shift(1)) & (close < ema50)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        signals = []

        if trend_bull.iloc[i] and cross_up.iloc[i]:
            sl = entry - atr_v * 2.0
            tp = entry + atr_v * 4.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": entry + atr_v * 0.5, "low": entry - atr_v * 0.5},
                "pattern_key": "macro_2ema_long",
                "strategy":    self.name,
                "notes":       f"EMA50={ema50.iloc[i]:.1f} > EMA200={ema200.iloc[i]:.1f}, cross up",
            })

        if trend_bear.iloc[i] and cross_down.iloc[i]:
            sl = entry + atr_v * 2.0
            tp = entry - atr_v * 4.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": entry + atr_v * 0.5, "low": entry - atr_v * 0.5},
                "pattern_key": "macro_2ema_short",
                "strategy":    self.name,
                "notes":       f"EMA50={ema50.iloc[i]:.1f} < EMA200={ema200.iloc[i]:.1f}, cross down",
            })

        return signals
