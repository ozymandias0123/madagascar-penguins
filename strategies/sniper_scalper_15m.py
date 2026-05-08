"""
strategies/sniper_scalper_15m.py
Sniper Scalping Bot 15M  (NQ / ES / Gold)

Logic
-----
  - Trend filter   : close > EMA200 (bull) / close < EMA200 (bear)
  - Entry trigger  : close crosses above EMA20 + RSI > 55  (long)
                     close crosses below EMA20 + RSI < 45  (short)
  - Fixed TP / SL  : configurable tick offsets applied to ATR for
                     portability across different instruments
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SniperScalper15M(BaseStrategy):

    name        = "SniperScalper15M"
    description = "EMA20/50/200 sniper cross + RSI filter — 15M scalper"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    RSI_LONG_MIN  = 55
    RSI_SHORT_MAX = 45
    # TP/SL expressed as ATR multiples (mirrors the fixed-tick intent)
    TP_ATR  = 2.0
    SL_ATR  = 1.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 205:
            return []

        close = df["close"]
        atr_s = df["atr"] if "atr" in df.columns else self._calc_atr(df, 14)

        ema20  = close.ewm(span=20,  adjust=False).mean()
        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        rsi    = self._rsi(close, 14)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        r     = float(rsi.iloc[i])

        bull = entry > float(ema200.iloc[i])
        bear = entry < float(ema200.iloc[i])

        # EMA20 crossover/crossunder
        cross_up   = (float(close.iloc[i - 1]) <= float(ema20.iloc[i - 1]) and
                      float(close.iloc[i])     >  float(ema20.iloc[i]))
        cross_down = (float(close.iloc[i - 1]) >= float(ema20.iloc[i - 1]) and
                      float(close.iloc[i])     <  float(ema20.iloc[i]))

        long_cond  = bull and cross_up   and r > self.RSI_LONG_MIN
        short_cond = bear and cross_down and r < self.RSI_SHORT_MAX

        signals = []

        if long_cond:
            sl = entry - atr_v * self.SL_ATR
            tp = entry + atr_v * self.TP_ATR
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": float(ema20.iloc[i]) + atr_v * 0.2,
                                "low":  float(ema20.iloc[i]) - atr_v * 0.2},
                "pattern_key": "sniper_long",
                "strategy":    self.name,
                "notes":       (f"Sniper long: above EMA200, cross EMA20, "
                                f"RSI={r:.1f}>{self.RSI_LONG_MIN}"),
            })

        if short_cond:
            sl = entry + atr_v * self.SL_ATR
            tp = entry - atr_v * self.TP_ATR
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     6.5,
                "zone":        {"high": float(ema20.iloc[i]) + atr_v * 0.2,
                                "low":  float(ema20.iloc[i]) - atr_v * 0.2},
                "pattern_key": "sniper_short",
                "strategy":    self.name,
                "notes":       (f"Sniper short: below EMA200, cross EMA20, "
                                f"RSI={r:.1f}<{self.RSI_SHORT_MAX}"),
            })

        return signals

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
