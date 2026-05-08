"""
strategies/lucky13_scalper.py
Lucky13 EMA Scalper (Enhanced)

EMA13 crossover + candle direction filter.
VWAP: price must be above (long) or below (short) VWAP.
HTF proxy: longer EMA as substitute for 5-min EMA13 (no multi-TF in df).
Volume filter: volume > 1.2× its 20-bar SMA.
% based SL/TP (not ATR).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class Lucky13Scalper(BaseStrategy):

    name        = "Lucky13Scalper"
    description = "EMA13 cross + VWAP + volume filter + HTF EMA proxy — 1-min scalper"
    version     = "1.0"

    PROFIT_PCT  = 0.015    # 1.5%
    STOP_PCT    = 0.0075   # 0.75%
    VOL_MULT    = 1.2
    HTF_SPAN    = 65       # proxy for 5-min EMA13 (13 × 5 bars)

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.HTF_SPAN + 10:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"] if "atr" in df.columns else close.diff().abs().rolling(14).mean()

        ema13     = close.ewm(span=13, adjust=False).mean()
        ema_htf   = close.ewm(span=self.HTF_SPAN, adjust=False).mean()   # HTF EMA13 proxy

        # VWAP (session VWAP approximation using typical price rolling weighted mean)
        typical  = (high + low + close) / 3
        vol_safe = volume.replace(0, 1)
        vwap     = (typical * vol_safe).rolling(50).sum() / vol_safe.rolling(50).sum()

        # Volume filter
        vol_sma    = volume.rolling(20).mean()
        vol_filter = volume > vol_sma * self.VOL_MULT

        # EMA13 crossover with candle direction
        bull_cross = (close.shift(1) <= ema13.shift(1)) & (close > ema13) & (close > df["open"])
        bear_cross = (close.shift(1) >= ema13.shift(1)) & (close < ema13) & (close < df["open"])

        i     = -2
        last  = df.iloc[i]
        entry = float(last["close"])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        vwap_v   = float(vwap.iloc[i])
        htf_ema  = float(ema_htf.iloc[i])
        ema13_v  = float(ema13.iloc[i])
        vol_ok   = bool(vol_filter.iloc[i])

        long_filter  = entry > vwap_v and htf_ema > ema13_v and vol_ok
        short_filter = entry < vwap_v and htf_ema < ema13_v and vol_ok

        signals = []

        if bull_cross.iloc[i] and long_filter:
            sl = entry * (1 - self.STOP_PCT)
            tp = entry * (1 + self.PROFIT_PCT)
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.5,
                "zone":        {"high": entry + atr_v * 0.15, "low": entry - atr_v * 0.15},
                "pattern_key": "lucky13_long",
                "strategy":    self.name,
                "notes":       (f"EMA13 bull cross, above VWAP={vwap_v:.1f}, "
                                f"HTF EMA OK, vol filter {'OK' if vol_ok else 'FAIL'}"),
            })

        if bear_cross.iloc[i] and short_filter:
            sl = entry * (1 + self.STOP_PCT)
            tp = entry * (1 - self.PROFIT_PCT)
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     5.5,
                "zone":        {"high": entry + atr_v * 0.15, "low": entry - atr_v * 0.15},
                "pattern_key": "lucky13_short",
                "strategy":    self.name,
                "notes":       (f"EMA13 bear cross, below VWAP={vwap_v:.1f}, "
                                f"HTF EMA OK, vol filter {'OK' if vol_ok else 'FAIL'}"),
            })

        return signals
