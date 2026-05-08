"""
strategies/svt_swing.py
SVT 30M Options Swing
MACD crossover + EMA20 filter + HTF RSI bias
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SVTSwing(BaseStrategy):

    name        = "SVTSwing"
    description = "MACD signal cross + EMA20 + HTF RSI bias filter"
    version     = "1.0"

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 60:
            return []

        close = df["close"]
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()
        rsi   = df["rsi"] if "rsi" in df.columns else self._rsi(close, 14)

        ema20 = close.ewm(span=20,  adjust=False).mean()
        ema50 = close.ewm(span=50,  adjust=False).mean()

        # MACD (12, 26, 9)
        ema12      = close.ewm(span=12, adjust=False).mean()
        ema26      = close.ewm(span=26, adjust=False).mean()
        macd_line  = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        macd_cross_up   = (macd_line.shift(1) <= signal_line.shift(1)) & (macd_line > signal_line)
        macd_cross_down = (macd_line.shift(1) >= signal_line.shift(1)) & (macd_line < signal_line)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        r     = float(rsi.iloc[i])
        e20   = float(ema20.iloc[i])
        e50   = float(ema50.iloc[i])

        # HTF bull proxy: above EMA50 + RSI > 50
        htf_bull = entry > e50 and r > 50
        htf_bear = entry < e50 and r < 50

        signals = []

        if htf_bull and macd_cross_up.iloc[i] and entry > e20:
            sl = entry - atr_v * 1.5
            tp = entry + atr_v * 3.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "svt_macd_long",
                "strategy":    self.name,
                "notes":       f"MACD bull cross, HTF bull (RSI={r:.1f}), above EMA20",
            })

        if htf_bear and macd_cross_down.iloc[i] and entry < e20:
            sl = entry + atr_v * 1.5
            tp = entry - atr_v * 3.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "svt_macd_short",
                "strategy":    self.name,
                "notes":       f"MACD bear cross, HTF bear (RSI={r:.1f}), below EMA20",
            })

        return signals

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
