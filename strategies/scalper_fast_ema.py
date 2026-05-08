"""
strategies/scalper_fast_ema.py
XAUUSD Scalping Strategy V2 (Aggressive)
EMA5/13 crossover + RSI + volatility filter
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class ScalperFastEMA(BaseStrategy):

    name        = "ScalperFastEMA"
    description = "EMA5/13 cross + RSI50 + volatility filter — aggressive scalper"
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
        rsi   = df["rsi"] if "rsi" in df.columns else self._rsi(close)

        ema5  = close.ewm(span=5,  adjust=False).mean()
        ema13 = close.ewm(span=13, adjust=False).mean()

        # volatility OK: current ATR > 80% of its 50-bar SMA
        vol_ok = atr_s > atr_s.rolling(50).mean() * 0.8

        cross_up   = (ema5.shift(1) <= ema13.shift(1)) & (ema5 > ema13)
        cross_down = (ema5.shift(1) >= ema13.shift(1)) & (ema5 < ema13)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        r     = float(rsi.iloc[i])
        bull_candle = last["close"] > last["open"]
        bear_candle = last["close"] < last["open"]

        signals = []

        if (cross_up.iloc[i] and r > 50
                and vol_ok.iloc[i] and bull_candle):
            sl = entry - atr_v * 1.0
            tp = entry + atr_v * 2.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.0,
                "zone":        {"high": entry + atr_v * 0.2, "low": entry - atr_v * 0.2},
                "pattern_key": "scalp_ema_long",
                "strategy":    self.name,
                "notes":       f"EMA5 cross EMA13 up, RSI={r:.1f}, vol OK",
            })

        if (cross_down.iloc[i] and r < 50
                and vol_ok.iloc[i] and bear_candle):
            sl = entry + atr_v * 1.0
            tp = entry - atr_v * 2.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.0,
                "zone":        {"high": entry + atr_v * 0.2, "low": entry - atr_v * 0.2},
                "pattern_key": "scalp_ema_short",
                "strategy":    self.name,
                "notes":       f"EMA5 cross EMA13 down, RSI={r:.1f}, vol OK",
            })

        return signals

    @staticmethod
    def _rsi(close: pd.Series, period: int = 7) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
