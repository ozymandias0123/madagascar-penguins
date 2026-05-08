"""
strategies/bb_sniper.py
BB Sniper (Mean Reversion)
Bollinger Band extreme + RSI + EMA200 trend + session filter
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class BBSniper(BaseStrategy):

    name        = "BBSniper"
    description = "BB lower/upper touch + RSI extreme + EMA200 + session filter"
    version     = "1.0"

    # London + NY sessions map to these UTC hours
    ACTIVE_SESSIONS = {"london", "new_york"}

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 210:
            return []

        # session filter (London + NY only)
        if session not in self.ACTIVE_SESSIONS:
            return []

        close = df["close"]
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()
        rsi   = df["rsi"] if "rsi" in df.columns else self._rsi(close, 14)

        ema200 = close.ewm(span=200, adjust=False).mean()

        # Bollinger Bands (20, 2)
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # crossover back into band
        cross_above_lower = (close.shift(1) <= bb_lower.shift(1)) & (close > bb_lower)
        cross_below_upper = (close.shift(1) >= bb_upper.shift(1)) & (close < bb_upper)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        r     = float(rsi.iloc[i])
        e200  = float(ema200.iloc[i])

        signals = []

        # Long: bullish trend, price touches lower BB, RSI oversold, crosses back up
        if (entry > e200 and r < 35
                and cross_above_lower.iloc[i]):
            sl = entry - atr_v * 1.5
            tp = entry + atr_v * 3.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": float(bb_lower.iloc[i]) + atr_v * 0.3,
                                "low":  float(bb_lower.iloc[i]) - atr_v * 0.3},
                "pattern_key": "bb_sniper_long",
                "strategy":    self.name,
                "notes":       f"BB lower touch, RSI={r:.1f}, above EMA200, session={session}",
            })

        # Short: bearish trend, price touches upper BB, RSI overbought, crosses back down
        if (entry < e200 and r > 65
                and cross_below_upper.iloc[i]):
            sl = entry + atr_v * 1.5
            tp = entry - atr_v * 3.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": float(bb_upper.iloc[i]) + atr_v * 0.3,
                                "low":  float(bb_upper.iloc[i]) - atr_v * 0.3},
                "pattern_key": "bb_sniper_short",
                "strategy":    self.name,
                "notes":       f"BB upper touch, RSI={r:.1f}, below EMA200, session={session}",
            })

        return signals

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
