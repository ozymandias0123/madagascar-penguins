"""
strategies/gxg_trend_engine.py
GXG Trend Engine  (v1.0)

Logic
-----
  Adaptive volatility-trail line:
    buffer   = sensFactor (3.0) × ATR(atrWindow)
    trail[k] =
      price > trail[k-1] AND prev_price > trail[k-1]  → max(trail[k-1], price − buffer)
      price < trail[k-1] AND prev_price < trail[k-1]  → min(trail[k-1], price + buffer)
      price > trail[k-1]                               → price − buffer
      else                                             → price + buffer

  fastTrigger = EMA(close, 1) = close (period-1 EMA is identity).
  Signal:
    Long:  close crossover  trail  (close[i-1] ≤ trail[i-1] and close[i] > trail[i])
           AND close > trail
    Short: close crossunder trail
           AND close < trail

  SL:  trail line at signal bar
  TP:  entry ± TP_MULT × buffer  (no fixed TP in original — advisory)
  Exit: opposite crossover (encoded in notes)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class GXGTrendEngine(BaseStrategy):

    name        = "GXGTrendEngine"
    description = "Adaptive ATR-trail crossover; long on bull-cross, short on bear-cross"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    SENS_FACTOR = 3.0    # sensFactor in Pine
    ATR_WINDOW  = 1      # atrWindow — ATR(1) = current bar's true range (no smoothing)
    TP_MULT     = 2.0    # advisory TP multiplier (no fixed TP in original)

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 10:
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = self._calc_atr_arr(df, self.ATR_WINDOW)

        trail = self._compute_trail(close, atr_s, self.SENS_FACTOR)

        i = len(df) - 1
        if np.isnan(trail[i]) or np.isnan(trail[i - 1]) or np.isnan(atr_s[i]):
            return []

        c_now  = float(close[i])
        c_prev = float(close[i - 1])
        t_now  = float(trail[i])
        t_prev = float(trail[i - 1])

        bull_cross = (c_prev <= t_prev) and (c_now > t_now)
        bear_cross = (c_prev >= t_prev) and (c_now < t_now)

        long_sig  = c_now > t_now and bull_cross
        short_sig = c_now < t_now and bear_cross

        if not long_sig and not short_sig:
            return []

        sig_type = "buy" if long_sig else "sell"
        entry    = c_now
        buf      = self.SENS_FACTOR * float(atr_s[i])

        # SL = trail line; widen slightly so it sits just beyond it
        sl = (min(t_now, entry - buf) if sig_type == "buy"
              else max(t_now, entry + buf))

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = (entry + risk * self.TP_MULT if sig_type == "buy"
              else entry - risk * self.TP_MULT)

        quality = self._quality(sig_type, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(t_now + buf, 5),
                            "low":  round(t_now - buf, 5)},
            "pattern_key": f"gxg_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"GXG trail cross {'up' if long_sig else 'down'} | "
                            f"trail={t_now:.5f} | buffer={buf:.5f} | "
                            f"exit=opposite_cross"),
        }]

    # ── Adaptive trail computation ────────────────────────────────────────────

    @staticmethod
    def _compute_trail(close: np.ndarray, atr: np.ndarray,
                       sens: float) -> np.ndarray:
        n     = len(close)
        trail = np.full(n, np.nan)
        trail[0] = close[0]
        for k in range(1, n):
            if np.isnan(atr[k]):
                trail[k] = trail[k - 1]
                continue
            buf   = sens * atr[k]
            price = close[k]
            prev  = trail[k - 1]
            pp    = close[k - 1]
            if price > prev and pp > prev:
                trail[k] = max(prev, price - buf)
            elif price < prev and pp < prev:
                trail[k] = min(prev, price + buf)
            elif price > prev:
                trail[k] = price - buf
            else:
                trail[k] = price + buf
        return trail

    # ── ATR — supports period=1 (raw true range) ─────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values;  l = df["low"].values;  c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        if n == 1:
            return tr          # ATR(1) = raw TR, no smoothing
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, context: dict, htf_bias: str) -> float:
        score = 5.5
        if context.get("adx", 0) > 25:
            score += 1.0
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 1.0
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 1.0
        if context.get("volatility", "normal") == "high":
            score -= 0.5
        return round(min(max(score, 1.0), 10.0), 1)
