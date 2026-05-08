"""
strategies/crypto_ncs.py
Crypto NCS v6
Z-score oversold + RSI < 30 + price above EMA8 — long-only
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class CryptoNCS(BaseStrategy):

    name        = "CryptoNCS"
    description = "Z-score < -2 + RSI < 30 + above EMA8 — long-only mean reversion"
    version     = "1.0"

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
        atr_s = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()
        rsi   = df["rsi"] if "rsi" in df.columns else self._rsi(close, 14)

        ema8    = close.ewm(span=8, adjust=False).mean()
        sma20   = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        z_score = (close - sma20) / std20.replace(0, np.nan)

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        z     = float(z_score.iloc[i])
        r     = float(rsi.iloc[i])
        e8    = float(ema8.iloc[i])

        signals = []

        # Long-only: deep z-score, oversold RSI, above EMA8
        if z < -2.0 and r < 30 and entry > e8:
            sl = entry - atr_v * 1.8
            tp = entry + atr_v * 4.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": entry + atr_v * 0.3, "low": entry - atr_v * 0.3},
                "pattern_key": "crypto_ncs_long",
                "strategy":    self.name,
                "notes":       f"Z={z:.2f} < -2, RSI={r:.1f} < 30, above EMA8",
            })

        return signals

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))
