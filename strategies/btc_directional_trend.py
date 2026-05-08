"""
strategies/btc_directional_trend.py
BTC Directional Trend + Trailing SL  (v1.0)

Logic
-----
  Trend filter:  EMA(200)
    uptrend   = close > EMA200
    downtrend = close < EMA200

  Entry:
    Long:  uptrend   AND close crosses OVER  EMA(20)
    Short: downtrend AND close crosses UNDER EMA(20)

  Trailing stop (advisory, encoded in notes):
    Long:  stop tracks at  close − ATR(14) × atrMultiplier
    Short: stop tracks at  close + ATR(14) × atrMultiplier

  SL (at entry bar):  entry ∓ ATR × atrMultiplier  (initial trail value)
  TP:  entry ± ATR × atrMultiplier × 3.0            (3× the SL distance — advisory)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class BTCDirectionalTrend(BaseStrategy):

    name        = "BTCDirectionalTrend"
    description = "EMA200 trend + EMA20 cross entry, ATR trailing stop"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    EMA_TREND   = 200
    EMA_ENTRY   = 20
    ATR_LEN     = 14
    ATR_MULT    = 2.0    # atrMultiplier
    TP_MULT     = 3.0    # advisory TP multiplier relative to ATR (not in original)

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.EMA_TREND + 5:
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        ema200 = self._ema(close, self.EMA_TREND)
        ema20  = self._ema(close, self.EMA_ENTRY)

        i = len(df) - 1

        if (np.isnan(ema200[i]) or np.isnan(ema20[i])
                or np.isnan(ema200[i-1]) or np.isnan(ema20[i-1])
                or np.isnan(atr_s[i])):
            return []

        c_now   = float(close[i])
        c_prev  = float(close[i - 1])
        e20_now  = float(ema20[i])
        e20_prev = float(ema20[i - 1])
        e200    = float(ema200[i])
        atr_val = float(atr_s[i])

        uptrend   = c_now > e200
        downtrend = c_now < e200

        cross_over  = c_prev <= e20_prev and c_now > e20_now
        cross_under = c_prev >= e20_prev and c_now < e20_now

        long_cond  = uptrend   and cross_over
        short_cond = downtrend and cross_under

        if not long_cond and not short_cond:
            return []

        sig_type = "buy" if long_cond else "sell"
        entry    = c_now

        sl = (entry - atr_val * self.ATR_MULT if sig_type == "buy"
              else entry + atr_val * self.ATR_MULT)
        tp = (entry + atr_val * self.TP_MULT  if sig_type == "buy"
              else entry - atr_val * self.TP_MULT)

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(sig_type, context, htf_bias, c_now, e200, atr_val)

        trail_init = sl   # trailing stop starts at SL level

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(e200 + atr_val, 5),
                            "low":  round(e200 - atr_val, 5)},
            "pattern_key": f"btc_dir_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"EMA{self.EMA_ENTRY} cross {'up' if long_cond else 'down'} | "
                            f"EMA{self.EMA_TREND}={e200:.2f} | "
                            f"trail_init={trail_init:.2f} | "
                            f"trail_step=ATR×{self.ATR_MULT}"),
        }]

    # ── EMA ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        out  = np.full(len(arr), np.nan)
        mult = 2.0 / (n + 1)
        if len(arr) < n:
            return out
        out[n - 1] = arr[:n].mean()
        for k in range(n, len(arr)):
            out[k] = arr[k] * mult + out[k - 1] * (1 - mult)
        return out

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values;  l = df["low"].values;  c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, context: dict, htf_bias: str,
                 price: float, ema200: float, atr: float) -> float:
        score = 5.5
        dist = abs(price - ema200) / (atr + 1e-10)
        if dist < 5:
            score += 0.5   # close to EMA200 — fresh breakout
        if context.get("adx", 0) > 25:
            score += 1.0
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 1.0
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)
