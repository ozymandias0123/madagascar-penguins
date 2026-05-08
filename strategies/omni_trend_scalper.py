"""
strategies/omni_trend_scalper.py
OmniTrend ATR Scalper  (v1.0)

Logic
-----
  Custom adaptive trend line (t_line) that steps toward price at a variable rate:

    v_atr  = ATR(len_atr=200)   — long-period ATR for scale

    Per bar:
      d = close − t_line
      sig_l = (t_dir == −1) AND d  >  mult_rev × v_atr   → reverse to bull
      sig_s = (t_dir == +1) AND −d >  mult_rev × v_atr   → reverse to bear
      if reversal: t_dir flips
      s_chk = |d| > mult_step × v_atr  →  fast mode if true
      if reversal, mode-change, or first step:
          t_step = v_atr / len_fast  (5) if fast else v_atr / len_slow (10)
      t_line += t_dir × t_step

  Signal:
    Long:  sig_l  (t_dir just flipped +1)
    Short: sig_s  (t_dir just flipped −1)

  SL:  entry ∓ v_atr × sl_mult  (2.0)
  TP:  entry ± v_atr × tp_mult  (3.0)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class OmniTrendScalper(BaseStrategy):

    name        = "OmniTrendScalper"
    description = "Custom ATR step-trend adaptive line; reversal signals"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    LEN_FAST  = 5
    LEN_SLOW  = 10
    LEN_ATR   = 200
    MULT_STEP = 1.0      # step multiplier — when fast mode activates
    MULT_REV  = 2.0      # reversal multiplier
    SL_MULT   = 2.0
    TP_MULT   = 3.0

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.LEN_ATR + 5:
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        sig_l, sig_s, t_line = self._compute_trend(
            close, atr_s, self.LEN_ATR, self.LEN_FAST, self.LEN_SLOW,
            self.MULT_STEP, self.MULT_REV)

        i = len(df) - 1
        if not sig_l[i] and not sig_s[i]:
            return []

        if np.isnan(atr_s[i]):
            return []

        sig_type = "buy" if sig_l[i] else "sell"
        entry    = float(close[i])
        atr_val  = float(atr_s[i])

        sl = (entry - atr_val * self.SL_MULT if sig_type == "buy"
              else entry + atr_val * self.SL_MULT)
        tp = (entry + atr_val * self.TP_MULT  if sig_type == "buy"
              else entry - atr_val * self.TP_MULT)

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(sig_type, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(float(t_line[i]) + atr_val, 5),
                            "low":  round(float(t_line[i]) - atr_val, 5)},
            "pattern_key": f"omni_trend_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"OmniTrend reversal {'long' if sig_l[i] else 'short'} | "
                            f"t_line={t_line[i]:.5f} | ATR={atr_val:.5f}"),
        }]

    # ── Adaptive trend line engine ────────────────────────────────────────────

    @staticmethod
    def _compute_trend(close:    np.ndarray,
                       atr:      np.ndarray,
                       len_atr:  int,
                       len_fast: int,
                       len_slow: int,
                       m_step:   float,
                       m_rev:    float):
        n      = len(close)
        t_line = np.full(n, np.nan)
        sig_l  = np.zeros(n, dtype=bool)
        sig_s  = np.zeros(n, dtype=bool)

        # Initialise at first valid ATR bar
        start = len_atr - 1
        if start >= n:
            return sig_l, sig_s, t_line

        t_line[start] = close[start]
        t_dir   = 1
        t_step  = 0.0
        is_fast = False

        for k in range(start + 1, n):
            if np.isnan(atr[k]):
                t_line[k] = t_line[k - 1]
                continue

            v_atr = atr[k]
            prev  = t_line[k - 1]
            if np.isnan(prev):
                t_line[k] = close[k]
                continue

            d     = close[k] - prev
            sl    = t_dir == -1 and d > m_rev * v_atr
            ss    = t_dir == 1  and -d > m_rev * v_atr

            if sl:
                t_dir = 1
                sig_l[k] = True
            elif ss:
                t_dir = -1
                sig_s[k] = True

            s_chk = abs(d) > m_step * v_atr
            if sl or ss or (s_chk != is_fast) or t_step == 0.0:
                is_fast = s_chk
                t_step  = v_atr / len_fast if is_fast else v_atr / len_slow

            t_line[k] = prev + t_dir * t_step

        return sig_l, sig_s, t_line

    # ── ATR (long period) ─────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 200) -> np.ndarray:
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
    def _quality(sig_type: str, context: dict, htf_bias: str) -> float:
        score = 5.5
        if context.get("adx", 0) > 25:
            score += 1.0
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 1.0
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)
