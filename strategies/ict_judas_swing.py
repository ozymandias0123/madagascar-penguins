"""
strategies/ict_judas_swing.py
ICT Judas Swing

Logic
-----
  The Judas Swing is a fake move at the opening of a session designed
  to trap retail traders, then reverse sharply in the true direction.

  Setup:
    1. At London or NY open (session = 'london' or 'new_york').
    2. In the first N bars of the session, price makes a swing in one
       direction (fake pump or dump) — we proxy this with the first
       bar's direction vs the previous session's range midpoint.
    3. If the open bar is strongly bullish (close > prev session mid):
       → expect Judas Swing HIGH to be swept, then reversal DOWN.
    4. If the open bar is strongly bearish:
       → expect Judas Swing LOW to be swept, then reversal UP.
    5. Reversal confirmation: current bar closes OPPOSITE to the Judas bar.
    6. FVG must be present after the reversal candle.

  Practical implementation (single-TF, current bar focus):
    - Swing detection: look at bar[-4] through bar[-1].
      If high was made at bar[-3] or [-2] and current bar closes
      below bar[-4] open → bearish reversal (judas was the rally).
      If low was made at bar[-3] or [-2] and current closes above bar[-4] open
      → bullish reversal.
    - FVG present in the reversal direction.
    - HTF bias confirms reversal direction.

  SL : 0.5×ATR beyond the Judas extreme (the fake high or low)
  TP : 2.5×risk
  Quality: 6 base, +1 session open, +1 FVG, +1 htf_bias, +1 ADX>20
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class ICTJudasSwing(BaseStrategy):

    name        = "ICTJudasSwing"
    description = "ICT Judas Swing: fake session open move + reversal FVG entry"
    version     = "1.0"

    SL_BUF = 0.5
    RR     = 2.5

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 20:
            return []

        if session not in ("london", "new_york"):
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        open_  = df["open"].values
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]) or i < 5:
            return []

        entry   = float(close[i])
        atr_val = float(atr_s[i])
        signals = []

        # Window: bars i-4 to i-1 for Judas detection
        # Judas HIGH scenario → bearish reversal
        window_high = float(np.max(high[i-4: i]))
        window_low  = float(np.min(low[i-4: i]))
        judas_high_bar = int(np.argmax(high[i-4: i])) + (i - 4)
        judas_low_bar  = int(np.argmin(low[i-4:  i])) + (i - 4)

        # ── Bearish Judas (fake rally → sell) ────────────────────────────────
        # Judas high was NOT the most recent bar, and current bar reversal down
        if judas_high_bar < i - 1:
            if float(close[i]) < float(open_[i]):   # current bar is bearish
                judas_extreme = window_high
                # FVG: bearish gap
                if i >= 3 and float(high[i-1]) < float(low[i-3]):
                    fvg_low  = float(high[i-1])
                    fvg_high = float(low[i-3])
                    sl   = judas_extreme + self.SL_BUF * atr_val
                    risk = abs(sl - entry)
                    if risk > 1e-10 and htf_bias in ("bearish", "neutral"):
                        tp   = entry - self.RR * risk
                        qual = self._quality(session, htf_bias, "sell", context)
                        signals.append({
                            "type":        "sell",
                            "entry_price": round(entry, 5),
                            "sl_price":    round(sl, 5),
                            "tp_price":    round(tp, 5),
                            "quality":     qual,
                            "zone":        {"high": round(fvg_high, 5),
                                            "low":  round(fvg_low, 5)},
                            "pattern_key": "ict_judas_swing_sell",
                            "strategy":    self.name,
                            "notes":       (f"Judas high={round(judas_extreme,5)} | "
                                            f"bear reversal | FVG | session={session}"),
                        })

        # ── Bullish Judas (fake dump → buy) ──────────────────────────────────
        if judas_low_bar < i - 1:
            if float(close[i]) > float(open_[i]):   # current bar is bullish
                judas_extreme = window_low
                if i >= 3 and float(low[i-1]) > float(high[i-3]):
                    fvg_low2  = float(high[i-3])
                    fvg_high2 = float(low[i-1])
                    sl   = judas_extreme - self.SL_BUF * atr_val
                    risk = abs(entry - sl)
                    if risk > 1e-10 and htf_bias in ("bullish", "neutral"):
                        tp   = entry + self.RR * risk
                        qual = self._quality(session, htf_bias, "buy", context)
                        signals.append({
                            "type":        "buy",
                            "entry_price": round(entry, 5),
                            "sl_price":    round(sl, 5),
                            "tp_price":    round(tp, 5),
                            "quality":     qual,
                            "zone":        {"high": round(fvg_high2, 5),
                                            "low":  round(fvg_low2, 5)},
                            "pattern_key": "ict_judas_swing_buy",
                            "strategy":    self.name,
                            "notes":       (f"Judas low={round(judas_extreme,5)} | "
                                            f"bull reversal | FVG | session={session}"),
                        })

        return signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(session, htf_bias, sig_type, context) -> float:
        score = 6.0
        score += 1.0   # session open verified
        score += 1.0   # FVG verified
        bias_ok = (sig_type == "buy"  and htf_bias == "bullish") or \
                  (sig_type == "sell" and htf_bias == "bearish")
        if bias_ok: score += 1.0
        if context.get("adx", 0) > 20: score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values; l = df["low"].values; c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr
