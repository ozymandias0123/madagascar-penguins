"""
strategies/zscore_mean_reversion.py
Z-Score Mean Reversion Pro
Z-score extremes + RSI confirmation + EMA50 trend filter
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class ZScoreMeanReversion(BaseStrategy):

    name        = "ZScoreMeanReversion"
    description = "Z-score < -2 / > 2 + RSI oversold/overbought + EMA50 filter"
    version     = "1.0"

    Z_LEN   = 20
    RSI_LEN = 14

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
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()
        rsi   = df["rsi"] if "rsi" in df.columns else self._rsi(close, self.RSI_LEN)

        mean    = close.rolling(self.Z_LEN).mean()
        std     = close.rolling(self.Z_LEN).std()
        z_score = (close - mean) / std.replace(0, np.nan)
        ema50   = close.ewm(span=50, adjust=False).mean()

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        z     = float(z_score.iloc[i])
        r     = float(rsi.iloc[i])
        e50   = float(ema50.iloc[i])

        signals = []

        # Long: oversold z-score, RSI < 30, price above EMA50
        if z < -2.0 and r < 30 and entry > e50:
            sl = entry - atr_v * 1.5
            tp = entry + atr_v * 2.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "zscore_oversold",
                "strategy":    self.name,
                "notes":       f"Z={z:.2f} < -2, RSI={r:.1f} < 30, above EMA50",
            })

        # Short: overbought z-score, RSI > 70, price below EMA50
        if z > 2.0 and r > 70 and entry < e50:
            sl = entry + atr_v * 1.5
            tp = entry - atr_v * 2.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "zscore_overbought",
                "strategy":    self.name,
                "notes":       f"Z={z:.2f} > 2, RSI={r:.1f} > 70, below EMA50",
            })

        return signals

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
