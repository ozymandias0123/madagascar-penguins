"""
strategies/smc_pro_ultimate.py
LuxAlgo SMC Pro Ultimate  (v1.0)

Logic
-----
  Premium / Discount zone:
    range_high = highest(high, pdLookback=100)
    range_low  = lowest(low,  pdLookback=100)
    equilibrium = (range_high + range_low) / 2
    discount = close < equilibrium   (buy side)
    premium  = close > equilibrium   (sell side)

  Fair Value Gaps (FVG):
    Bullish FVG: low[i] > high[i-2] AND close[i-1] > high[i-2]
    Bearish FVG: high[i] < low[i-2] AND close[i-1] < low[i-2]
    Trigger checks bFVG[i-1] OR bFVG[i].

  Market Structure Shift (MSS):
    Internal: pivot over ±internalLookback (9) bars
    Swing:    pivot over ±swingLookback    (50) bars
    MSS_Long  = close crosses above lastPivotHigh  (internal or swing, per mode)
    MSS_Short = close crosses below lastPivotLow

  Volume momentum: volume of last volCandles (3) bars all increasing.

  Entry (structureMode="All"):
    Buy:  mssL AND (discount if requirePDZone) AND (FVG if useFVGConfluence)
    Sell: mssS AND (premium  if requirePDZone) AND (FVG if useFVGConfluence)
    Optional divergence / BB filters (disabled by default).

  SL:  low  − ATR(9) × 3.0  (long)
  TP1: entry + (entry − SL) × 1.5  (50% close; encoded in notes)
  TP (engine target): TP1 level (full close advisory)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from strategies.base_strategy import BaseStrategy


class SMCProUltimate(BaseStrategy):

    name        = "SMCProUltimate"
    description = "MSS breakout + PD zone + FVG confluence, ATR SL, partial TP1"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    INTERNAL_LB      = 9
    SWING_LB         = 50
    STRUCTURE_MODE   = "All"      # "Internal" | "Swing" | "All"
    ATR_LEN          = 9
    ATR_MULT         = 3.0
    TP1_RR           = 1.5
    PD_LOOKBACK      = 100
    VOL_CANDLES      = 3
    VOL_MULT         = 1.2

    REQUIRE_PD_ZONE  = True
    USE_FVG          = True
    USE_DIV          = False
    USE_BB           = False

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.SWING_LB * 2 + self.PD_LOOKBACK + 5
        if len(df) < needed:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        atr_val = float(atr_s[i])
        c_now   = float(close[i])
        c_prev  = float(close[i - 1])

        # ── Premium / Discount zone ───────────────────────────────────────────
        lb         = min(self.PD_LOOKBACK, i)
        range_high = float(np.max(high[i - lb: i + 1]))
        range_low  = float(np.min(low[i  - lb: i + 1]))
        equil      = (range_high + range_low) / 2.0
        in_discount = c_now < equil
        in_premium  = c_now > equil

        # ── Fair Value Gaps ───────────────────────────────────────────────────
        def bull_fvg(k):
            return k >= 2 and low[k] > high[k - 2] and close[k - 1] > high[k - 2]

        def bear_fvg(k):
            return k >= 2 and high[k] < low[k - 2] and close[k - 1] < low[k - 2]

        b_fvg = bull_fvg(i) or bull_fvg(i - 1)
        s_fvg = bear_fvg(i) or bear_fvg(i - 1)

        # ── Pivot detection ───────────────────────────────────────────────────
        last_ish_i, last_isl_i = self._find_pivots(high, low, i, self.INTERNAL_LB)
        last_ssh_s, last_ssl_s = self._find_pivots(high, low, i, self.SWING_LB)

        def mss_long():
            crosses = []
            if self.STRUCTURE_MODE in ("Internal", "All") and last_ish_i is not None:
                crosses.append(c_prev <= last_ish_i and c_now > last_ish_i)
            if self.STRUCTURE_MODE in ("Swing", "All") and last_ssh_s is not None:
                crosses.append(c_prev <= last_ssh_s and c_now > last_ssh_s)
            return any(crosses)

        def mss_short():
            crosses = []
            if self.STRUCTURE_MODE in ("Internal", "All") and last_isl_i is not None:
                crosses.append(c_prev >= last_isl_i and c_now < last_isl_i)
            if self.STRUCTURE_MODE in ("Swing", "All") and last_ssl_s is not None:
                crosses.append(c_prev >= last_ssl_s and c_now < last_ssl_s)
            return any(crosses)

        mss_l = mss_long()
        mss_s = mss_short()

        if not mss_l and not mss_s:
            return []

        # ── Volume momentum ───────────────────────────────────────────────────
        vol_ok = True
        vc = min(self.VOL_CANDLES, i)
        for v in range(vc):
            if volume[i - v] < volume[i - v - 1]:
                vol_ok = False
                break
        vol_avg = float(np.mean(volume[max(0, i - 20): i])) if i > 20 else 1.0
        strong  = float(volume[i]) > vol_avg * self.VOL_MULT and vol_ok

        # ── Trigger conditions ────────────────────────────────────────────────
        b_trigger = (mss_l
                     and (not self.REQUIRE_PD_ZONE or in_discount)
                     and (not self.USE_FVG          or b_fvg))
        s_trigger = (mss_s
                     and (not self.REQUIRE_PD_ZONE or in_premium)
                     and (not self.USE_FVG          or s_fvg))

        if not b_trigger and not s_trigger:
            return []

        sig_type = "buy" if b_trigger else "sell"
        entry    = c_now

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl  = float(low[i]) - atr_val * self.ATR_MULT
        else:
            sl  = float(high[i]) + atr_val * self.ATR_MULT

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp1 = (entry + risk * self.TP1_RR if sig_type == "buy"
               else entry - risk * self.TP1_RR)

        quality = self._quality(sig_type, strong, in_discount, in_premium,
                                context, htf_bias)

        label = ("STRONG SMC " if strong else "SMC ") + sig_type.upper()

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp1, 5),
            "quality":     quality,
            "zone":        {"high": round(equil + atr_val, 5),
                            "low":  round(equil - atr_val, 5)},
            "pattern_key": f"smc_pro_{sig_type}_{'strong' if strong else 'normal'}",
            "strategy":    self.name,
            "notes":       (f"{label} [MSS+{'DISCOUNT' if sig_type=='buy' else 'PREMIUM'}] | "
                            f"FVG={'yes' if (b_fvg if sig_type=='buy' else s_fvg) else 'no'} | "
                            f"TP1={tp1:.2f} (50%@{self.TP1_RR}R) | "
                            f"trail=ATR×{self.ATR_MULT} after TP1"),
        }]

    # ── Pivot search ─────────────────────────────────────────────────────────

    @staticmethod
    def _find_pivots(high: np.ndarray, low: np.ndarray,
                     current: int, lookback: int
                     ) -> Tuple[Optional[float], Optional[float]]:
        """Return (last_pivot_high, last_pivot_low) scanning back from current."""
        last_ph = last_pl = None
        window  = lookback * 2 + 1
        # scan recent bars to find last confirmed pivot
        scan_start = current - lookback
        for k in range(scan_start, max(lookback, scan_start - 200), -1):
            if k < lookback or k + lookback >= len(high):
                break
            seg_h = high[k - lookback: k + lookback + 1]
            seg_l = low[k  - lookback: k + lookback + 1]
            if len(seg_h) < window:
                continue
            if last_ph is None and high[k] == np.max(seg_h):
                last_ph = float(high[k])
            if last_pl is None and low[k] == np.min(seg_l):
                last_pl = float(low[k])
            if last_ph is not None and last_pl is not None:
                break
        return last_ph, last_pl

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 9) -> np.ndarray:
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
    def _quality(sig_type: str, strong: bool, in_discount: bool,
                 in_premium: bool, context: dict, htf_bias: str) -> float:
        score = 5.5
        if strong:
            score += 1.5
        if sig_type == "buy"  and in_discount:
            score += 0.5
        elif sig_type == "sell" and in_premium:
            score += 0.5
        if context.get("adx", 0) > 25:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
