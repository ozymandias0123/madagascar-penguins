"""
strategies/ema_slope_rsi_osc.py
EMA Slope-RSI Oscillator  (v1.0)

Logic
-----
  EMA slope (arctan-normalized):
    raw_slope  = EMA(close, emaLength=20) − EMA(close, emaLength=20)[1]
    maDFScale  = stdev(raw_slope, scaleLen=100) × scaleMultiplier (1.0)
    maDf       = (100 / π) × atan(raw_slope / maDFScale)
    Range: asymptotically ±50; values near ±50 = very steep slope.

  NTZ (No-Trade Zone) boundaries:  hLineHeight = 8.0, lLineHeight = -8.0
    In NTZ: |maDf| < hLineHeight  (between -8 and +8)

  Centered RSI:
    rsi_raw  = RSI(close, rsiLength=14)
    rsi_cent = rsi_raw − 50         (range ≈ -50..+50; 0 = neutral)

  Acceleration / delta check:
    delta = maDf − maDf[1]
    deltaSufficient = |delta| > deltaThreshold (0.5)

  Stretch filter (optional, enabled by default):
    Long:  maDf < stretchThresholdLong  (25.0)   — don't buy overextended upslope
    Short: maDf > stretchThresholdShort (-25.0)

  Primary signals (NTZ cross):
    crossAboveNTZ = maDf >  hLineHeight AND maDf[-1] <= hLineHeight
    crossBelowNTZ = maDf < lLineHeight  AND maDf[-1] >= lLineHeight

    long_primary  = crossAboveNTZ AND deltaSufficient AND stretchFilterLong
    short_primary = crossBelowNTZ AND deltaSufficient AND stretchFilterShort

  Acceleration signal (secondary, slope already outside NTZ):
    accel_long  = maDf > hLineHeight AND rsi_cent > 0 AND delta > deltaThreshold
    accel_short = maDf < lLineHeight AND rsi_cent < 0 AND delta < -deltaThreshold

  Entry: primary OR acceleration signal.

  SL:
    Long:  close − ATR(14) × atrmultiplier (4.4)
    Short: close + ATR(14) × atrmultiplier

  TP (advisory):
    Long:  close + ATR(14) × atrmultiplier × tpMult (1.5)
    Short: close − ATR(14) × atrmultiplier × tpMult

  Trailing stop: when |maDf| drops back inside NTZ (encoded in notes).
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class EMASlopeRSIOsc(BaseStrategy):

    name        = "EMASlopeRSIOsc"
    description = "Arctan-normalized EMA slope oscillator; NTZ cross entry + RSI confluence"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    EMA_LEN            = 20
    SCALE_LEN          = 100
    SCALE_MULT         = 1.0
    H_LINE             = 8.0      # upper NTZ boundary
    L_LINE             = -8.0     # lower NTZ boundary
    DELTA_THRESH       = 0.5
    STRETCH_LONG       = 25.0     # don't long if slope > this
    STRETCH_SHORT      = -25.0    # don't short if slope < this
    RSI_LEN            = 14
    ATR_LEN            = 14
    ATR_MULT           = 4.4
    TP_MULT            = 1.5

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.EMA_LEN + self.SCALE_LEN + self.RSI_LEN + 10
        if len(df) < needed:
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        # ── EMA slope ─────────────────────────────────────────────────────────
        ema_arr = self._ema(close, self.EMA_LEN)
        if np.isnan(ema_arr[i]) or np.isnan(ema_arr[i - 1]):
            return []

        raw_slope_arr = np.diff(ema_arr, prepend=ema_arr[0])
        # stdev of slope over scaleLen
        scale_start = max(0, i - self.SCALE_LEN + 1)
        slope_seg   = raw_slope_arr[scale_start: i + 1]
        valid_seg   = slope_seg[~np.isnan(slope_seg)]
        if len(valid_seg) < 5:
            return []
        ma_df_scale = float(np.std(valid_seg)) * self.SCALE_MULT + 1e-10

        def arctan_norm(s):
            return (100.0 / np.pi) * np.arctan(s / ma_df_scale)

        ma_df_now  = arctan_norm(float(raw_slope_arr[i]))
        ma_df_prev = arctan_norm(float(raw_slope_arr[i - 1]))

        delta = ma_df_now - ma_df_prev

        # ── Centered RSI ──────────────────────────────────────────────────────
        rsi_arr  = self._rsi(close, self.RSI_LEN)
        if np.isnan(rsi_arr[i]):
            return []
        rsi_cent = float(rsi_arr[i]) - 50.0

        # ── Signals ───────────────────────────────────────────────────────────
        delta_ok = abs(delta) > self.DELTA_THRESH

        cross_above_ntz = ma_df_now > self.H_LINE and ma_df_prev <= self.H_LINE
        cross_below_ntz = ma_df_now < self.L_LINE and ma_df_prev >= self.L_LINE

        stretch_ok_long  = ma_df_now < self.STRETCH_LONG
        stretch_ok_short = ma_df_now > self.STRETCH_SHORT

        long_primary  = cross_above_ntz and delta_ok and stretch_ok_long
        short_primary = cross_below_ntz and delta_ok and stretch_ok_short

        accel_long  = (ma_df_now > self.H_LINE and rsi_cent > 0
                       and delta > self.DELTA_THRESH and stretch_ok_long)
        accel_short = (ma_df_now < self.L_LINE and rsi_cent < 0
                       and delta < -self.DELTA_THRESH and stretch_ok_short)

        long_ok  = long_primary  or accel_long
        short_ok = short_primary or accel_short

        if not long_ok and not short_ok:
            return []

        sig_type = "buy" if long_ok else "sell"
        entry    = float(close[i])
        atr_val  = float(atr_s[i])

        sl = (entry - atr_val * self.ATR_MULT if sig_type == "buy"
              else entry + atr_val * self.ATR_MULT)
        tp = (entry + atr_val * self.ATR_MULT * self.TP_MULT if sig_type == "buy"
              else entry - atr_val * self.ATR_MULT * self.TP_MULT)

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        sig_class = "NTZ_cross" if (long_primary or short_primary) else "accel"
        quality   = self._quality(sig_type, sig_class, abs(ma_df_now), context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(entry + atr_val, 5),
                            "low":  round(entry - atr_val, 5)},
            "pattern_key": f"ema_slope_{sig_type}_{sig_class}",
            "strategy":    self.name,
            "notes":       (f"slope={ma_df_now:.2f} | delta={delta:.3f} | "
                            f"RSI_cent={rsi_cent:.1f} | "
                            f"signal={sig_class} | "
                            f"trail=exit_when_maDf_re-enters_NTZ"),
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

    # ── RSI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(close), np.nan)
        if len(close) < n + 1:
            return out
        delta = np.diff(close)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = gain[:n].mean()
        avg_l = loss[:n].mean()
        out[n] = 100 - 100 / (1 + avg_g / (avg_l + 1e-10))
        for k in range(n + 1, len(close)):
            avg_g = (avg_g * (n - 1) + gain[k - 1]) / n
            avg_l = (avg_l * (n - 1) + loss[k - 1]) / n
            out[k] = 100 - 100 / (1 + avg_g / (avg_l + 1e-10))
        return out

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h = df["high"].values; l = df["low"].values; c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, sig_class: str, slope_abs: float,
                 context: dict, htf_bias: str) -> float:
        score = 5.5
        if sig_class == "NTZ_cross":
            score += 0.5       # primary signal bonus
        if slope_abs > 20:
            score += 0.5       # strong momentum
        if context.get("adx", 0) > 25:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
