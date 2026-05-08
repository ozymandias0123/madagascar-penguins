"""
strategies/sma200_momentum.py
SMA200 Momentum

Logic
-----
  SMA200 defines the broad trend baseline.

  Buy conditions (all required):
    - close > SMA200 × 1.005  (at least +0.5% above)
    - Previous candle also closed above SMA200 (trend continuation)
    - RSI > 50 and RSI < 70

  Sell conditions (all required):
    - close < SMA200 × 0.995  (at least −0.5% below)
    - Previous candle also closed below SMA200 (trend continuation)
    - RSI < 50 and RSI > 30

  SL : 2×ATR below entry (buy) | 2×ATR above entry (sell)
  TP : 3×ATR from entry

  Quality (base 6):
    +1  ADX > 20
    +1  htf_bias matches
    +1  RSI in sweet spot (55–65 for buy, 35–45 for sell)
    −1  RSI > 65 for buy (risk of overbought)  |  RSI < 35 for sell
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy


class SMA200Momentum(BaseStrategy):

    name        = "SMA200Momentum"
    description = "SMA200 trend baseline with RSI momentum filter"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    SMA_PERIOD   = 200
    OFFSET_PCT   = 0.005      # 0.5 % minimum distance from SMA200
    RSI_BUY_LO   = 50;  RSI_BUY_HI  = 70
    RSI_SELL_LO  = 30;  RSI_SELL_HI = 50
    ATR_SL       = 2.0
    ATR_TP       = 3.0
    MIN_BARS     = 205

    # ── main entry point ─────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.MIN_BARS:
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1

        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else float(close[i]) * 0.001
        if atr_val <= 0:
            return []

        # ── SMA200 ────────────────────────────────────────────────────────────
        sma200_arr = self._sma(close, self.SMA_PERIOD)
        sma200     = float(sma200_arr[i])
        sma200_p   = float(sma200_arr[i - 1])   # previous bar

        if np.isnan(sma200) or np.isnan(sma200_p):
            return []

        entry      = float(close[i])
        close_prev = float(close[i - 1])

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_val = None
        if "rsi" in df.columns:
            rv = df["rsi"].values[i]
            if not np.isnan(rv):
                rsi_val = float(rv)

        if rsi_val is None:
            return []

        # ── Check buy conditions ──────────────────────────────────────────────
        buy_ok = (
            entry      > sma200   * (1.0 + self.OFFSET_PCT)
            and close_prev > sma200_p
            and self.RSI_BUY_LO < rsi_val < self.RSI_BUY_HI
        )

        # ── Check sell conditions ─────────────────────────────────────────────
        sell_ok = (
            entry      < sma200   * (1.0 - self.OFFSET_PCT)
            and close_prev < sma200_p
            and self.RSI_SELL_LO < rsi_val < self.RSI_SELL_HI
        )

        if not buy_ok and not sell_ok:
            return []

        sig_type = "buy" if buy_ok else "sell"

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl = entry - self.ATR_SL * atr_val
            tp = entry + self.ATR_TP * atr_val
        else:
            sl = entry + self.ATR_SL * atr_val
            tp = entry - self.ATR_TP * atr_val

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        # ── Quality ───────────────────────────────────────────────────────────
        quality = self._score_quality(sig_type, rsi_val, htf_bias, context)

        notes = (
            f"SMA200={round(sma200, 5)} | "
            f"close vs SMA200: {round((entry / sma200 - 1) * 100, 3)}% | "
            f"RSI={round(rsi_val, 1)} | "
            f"ATR={round(atr_val, 5)}"
        )

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {
                "high": round(float(df["high"].values[i]), 5),
                "low":  round(float(df["low"].values[i]),  5),
            },
            "pattern_key": f"sma200_momentum_{sig_type}",
            "strategy":    self.name,
            "notes":       notes,
        }]

    # ── quality scoring ───────────────────────────────────────────────────────

    @staticmethod
    def _score_quality(
        sig_type: str,
        rsi_val: float,
        htf_bias: str,
        context: Dict[str, Any],
    ) -> float:
        score = 6.0

        # ADX > 20
        if context.get("adx", 0) > 20:
            score += 1.0

        # HTF bias
        bias = context.get("htf_bias", htf_bias)
        if sig_type == "buy"  and bias == "bullish":
            score += 1.0
        if sig_type == "sell" and bias == "bearish":
            score += 1.0

        # RSI sweet spot
        if sig_type == "buy":
            if 55.0 <= rsi_val <= 65.0:
                score += 1.0
            if rsi_val > 65.0:        # approaching overbought
                score -= 1.0
        else:
            if 35.0 <= rsi_val <= 45.0:
                score += 1.0
            if rsi_val < 35.0:        # approaching oversold
                score -= 1.0

        return round(min(max(score, 1.0), 10.0), 1)

    # ── shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _sma(arr: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        if len(arr) < n:
            return out
        cumsum = np.cumsum(arr)
        out[n - 1:] = (cumsum[n - 1:] - np.concatenate([[0], cumsum[: -n]])) / n
        return out

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        c1 = np.roll(c, 1);  c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        if len(tr) >= n:
            atr[n - 1] = tr[:n].mean()
            for k in range(n, len(tr)):
                atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr
