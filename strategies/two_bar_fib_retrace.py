"""
strategies/two_bar_fib_retrace.py
Two-Bar Fibonacci Retrace Strategy [Futures]  (v1.0)

Logic
-----
  Two-bar impulse detection (bars i-2 = bar1, i-1 = bar2):
    Full body:  |close − open| / (high − low)  ≥  bodyPctMin (0.50)
    Vol confirm: volume[i-1] > volume[i-2]

    Bullish impulse — same direction:
      bar1 AND bar2 both bullish, both full body, vol confirm
    Bullish impulse — mixed:
      bar2.close > bar1.high  (engulf/gap), both full body, vol confirm
    Bearish mirrors above.

  VWAP filter (optional, default enabled):
    Bull: close > session-VWAP
    Bear: close < session-VWAP

  Entry: impulse AND current bar's wick enters the 50–61.8% Fibonacci
    retrace zone of bar2:
    Bull zone: [bar2.high − 61.8%×bar2.range,  bar2.high − 50%×bar2.range]
    Bear zone: [bar2.low  + 50%×bar2.range,    bar2.low  + 61.8%×bar2.range]

  SL:  bar2's opposite extreme − ATR × SL_BUF_MULT  (long)
       bar2's opposite extreme + ATR × SL_BUF_MULT  (short)
  TP:  entry ± 2× risk  (original has SL-only exit; 2R is advisory)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class TwoBarFibRetrace(BaseStrategy):

    name        = "TwoBarFibRetrace"
    description = "Two-bar impulse + 50-61.8% Fib wick retrace entry"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    BODY_PCT_MIN = 0.50   # minimum body-to-range ratio
    SL_BUF_MULT  = 0.3    # ATR buffer beyond bar2's extreme for SL
    RR_RATIO     = 2.0    # TP = risk × RR  (advisory)
    USE_VWAP     = True

    FIB_50  = 0.500
    FIB_618 = 0.618

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 30:
            return []

        close  = df["close"].values
        open_  = df["open"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1

        if np.isnan(atr_s[i]):
            return []

        # ── Bar definitions ───────────────────────────────────────────────────
        b1o, b1c, b1h, b1l = open_[i-2], close[i-2], high[i-2], low[i-2]
        b2o, b2c, b2h, b2l = open_[i-1], close[i-1], high[i-1], low[i-1]

        b1_rng = b1h - b1l
        b2_rng = b2h - b2l

        b1_full = (b1_rng > 0 and abs(b1c - b1o) / b1_rng >= self.BODY_PCT_MIN)
        b2_full = (b2_rng > 0 and abs(b2c - b2o) / b2_rng >= self.BODY_PCT_MIN)

        b1_bull = b1c > b1o
        b2_bull = b2c > b2o

        vol_confirm = float(volume[i-1]) > float(volume[i-2])

        # ── Impulse detection ─────────────────────────────────────────────────
        bull_same  = b1_bull and b2_bull and b1_full and b2_full and vol_confirm
        bull_mixed = (not (b1_bull and b2_bull)) and (b2c > b1h) and b1_full and b2_full and vol_confirm
        bear_same  = (not b1_bull) and (not b2_bull) and b1_full and b2_full and vol_confirm
        bear_mixed = (not (not b1_bull and not b2_bull)) and (b2c < b1l) and b1_full and b2_full and vol_confirm

        impulse_bull = bull_same or bull_mixed
        impulse_bear = bear_same or bear_mixed

        if not impulse_bull and not impulse_bear:
            return []

        # ── VWAP filter ───────────────────────────────────────────────────────
        if self.USE_VWAP:
            vwap = self._calc_vwap(df, close, high, low, volume)
            vwap_now = float(vwap[i])
            if np.isnan(vwap_now):
                pass   # no time data — skip filter
            else:
                if impulse_bull and close[i] <= vwap_now:
                    return []
                if impulse_bear and close[i] >= vwap_now:
                    return []

        # ── Retrace zone wick test ────────────────────────────────────────────
        atr_val = float(atr_s[i])
        c_now   = float(close[i])
        l_now   = float(low[i])
        h_now   = float(high[i])

        bull_zt = b2h - b2_rng * self.FIB_50
        bull_zb = b2h - b2_rng * self.FIB_618
        bear_zt = b2l + b2_rng * self.FIB_618
        bear_zb = b2l + b2_rng * self.FIB_50

        bull_sig = impulse_bull and (l_now <= bull_zt) and (l_now >= bull_zb)
        bear_sig = impulse_bear and (h_now >= bear_zb) and (h_now <= bear_zt)

        if not bull_sig and not bear_sig:
            return []

        sig_type  = "buy" if bull_sig else "sell"
        entry     = c_now
        imp_label = "Match" if (bull_same if bull_sig else bear_same) else "Mixed"

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl = b2l - self.SL_BUF_MULT * atr_val
        else:
            sl = b2h + self.SL_BUF_MULT * atr_val

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = (entry + risk * self.RR_RATIO if sig_type == "buy"
              else entry - risk * self.RR_RATIO)

        quality = self._quality(sig_type, imp_label, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(bull_zt if sig_type == "buy" else bear_zt, 5),
                            "low":  round(bull_zb if sig_type == "buy" else bear_zb, 5)},
            "pattern_key": f"two_bar_fib_{sig_type}_{imp_label.lower()}",
            "strategy":    self.name,
            "notes":       (f"Impulse [{imp_label}] | "
                            f"b2=[{b2l:.2f}-{b2h:.2f}] | "
                            f"fib_zone=[{(bull_zb if sig_type=='buy' else bear_zb):.2f}"
                            f"-{(bull_zt if sig_type=='buy' else bear_zt):.2f}] | "
                            f"exit=SL_only_orig"),
        }]

    # ── VWAP (session-resetting) ──────────────────────────────────────────────

    @staticmethod
    def _calc_vwap(df: pd.DataFrame, close, high, low, volume) -> np.ndarray:
        hlc3   = (high + low + close) / 3.0
        vwap   = np.full(len(close), np.nan)
        cum_pv = 0.0;  cum_v = 0.0;  prev_date = None
        dates  = None
        if hasattr(df.index, "date"):
            try:
                dates = [d.date() for d in df.index]
            except Exception:
                pass
        for k in range(len(close)):
            cur_date = dates[k] if dates is not None else None
            if cur_date != prev_date:
                cum_pv = 0.0;  cum_v = 0.0;  prev_date = cur_date
            v = float(volume[k]) if volume[k] > 0 else 1.0
            cum_pv += float(hlc3[k]) * v;  cum_v += v
            vwap[k] = cum_pv / cum_v
        return vwap

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
    def _quality(sig_type: str, imp_label: str, context: dict,
                 htf_bias: str) -> float:
        score = 5.5
        if imp_label == "Match":
            score += 1.0      # same-direction impulse is cleaner
        if context.get("adx", 0) > 20:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
