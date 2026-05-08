"""
strategies/sovereign_trend.py
Sovereign Trend Strategy [JOAT]  (v2.0 — faithful Pine port)

Logic
-----
  SMEMA (double-smoothed EMA):  SMEMA(src, len) = SMA( EMA(src, len), len )
  Signal: SMEMA_fast crosses SMEMA_slow

  Pine defaults (intentionally ultra-fast to maximise crossover frequency):
    smFast   = 2   (SMEMA fast period)
    smSlow   = 5   (SMEMA slow period)
    smBase   = 15  (baseline SMEMA — price-above/below filter)

  Optional confirmation filters — ALL DISABLED by default in Pine:
    - ADX > 18       (USE_ADX   = False)
    - RSI 52–70 long / 30–48 short  (USE_RSI = False)
    - Volume > 1.0× 20-bar avg      (USE_VOL = False)
    - Baseline SMEMA filter         (USE_BASELINE = False)

  TP structure:
    TP1 (50% partial):  2.5 × ATR from entry  → SL moves to break-even
    TP2 (full close):   4.5 × ATR from entry

  Trailing stop after TP1: engine moves SL to entry (encoded in notes).

  Reversal exit: fast SMEMA crosses back the other way (encoded in notes).

  Max-bars exit: if TP1 not reached within MAX_BARS (10) candles,
    signal is invalidated (engine awareness via notes).

  SL: 1.0 × ATR below/above entry
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SovereignTrend(BaseStrategy):

    name        = "SovereignTrend"
    description = "SMEMA(2/5) crossover; optional ADX/RSI/vol filters (off by default)"
    version     = "2.0"

    # ── parameters (matching Pine defaults exactly) ──────────────────────────
    FAST_LEN      = 2        # smFast  = 2
    SLOW_LEN      = 5        # smSlow  = 5
    BASE_LEN      = 15       # smBase  = 15  (baseline SMEMA)
    ADX_LEN       = 14
    ADX_MIN       = 18.0
    RSI_LEN       = 14
    RSI_LONG_MIN  = 52.0     # long: RSI must be > 52
    RSI_LONG_MAX  = 70.0
    RSI_SHORT_MIN = 30.0
    RSI_SHORT_MAX = 48.0     # short: RSI must be < 48
    VOL_LEN       = 20
    VOL_MULT      = 1.0
    ATR_SL_MULT   = 1.0
    TP1_ATR_MULT  = 2.5
    TP2_ATR_MULT  = 4.5
    MAX_BARS      = 10       # invalidate if TP1 not hit in this many bars

    # toggle filters — ALL disabled by default (matches Pine input.bool(false, ...))
    USE_ADX      = False
    USE_RSI      = False
    USE_VOL      = False
    USE_BASELINE = False

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.BASE_LEN * 2 + self.ADX_LEN + 5
        if len(df) < needed:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns \
                 else np.ones(len(df))
        atr_s  = df["atr"].values if "atr" in df.columns \
                 else self._calc_atr_arr(df)

        # ── SMEMA ─────────────────────────────────────────────────────────────
        smema_fast = self._smema(close, self.FAST_LEN)
        smema_slow = self._smema(close, self.SLOW_LEN)
        smema_base = self._smema(close, self.BASE_LEN)

        i = len(df) - 1

        if np.isnan(smema_fast[i]) or np.isnan(smema_slow[i]):
            return []
        if np.isnan(smema_fast[i - 1]) or np.isnan(smema_slow[i - 1]):
            return []

        # ── Cross detection ───────────────────────────────────────────────────
        cross_up   = (smema_fast[i - 1] <= smema_slow[i - 1] and
                      smema_fast[i]     >  smema_slow[i])
        cross_down = (smema_fast[i - 1] >= smema_slow[i - 1] and
                      smema_fast[i]     <  smema_slow[i])

        if not cross_up and not cross_down:
            return []

        sig_type = "buy" if cross_up else "sell"
        entry    = float(close[i])
        atr_val  = float(atr_s[i]) if not np.isnan(atr_s[i]) else entry * 0.001

        # ── Filters ───────────────────────────────────────────────────────────
        if self.USE_ADX:
            adx = context.get("adx") or self._adx(high, low, close, i)
            if adx < self.ADX_MIN:
                return []

        if self.USE_RSI:
            rsi_arr = self._rsi(close, self.RSI_LEN)
            rsi_val = float(rsi_arr[i])
            if not np.isnan(rsi_val):
                if sig_type == "buy" and not (self.RSI_LONG_MIN <= rsi_val <= self.RSI_LONG_MAX):
                    return []
                if sig_type == "sell" and not (self.RSI_SHORT_MIN <= rsi_val <= self.RSI_SHORT_MAX):
                    return []
            else:
                rsi_val = 50.0
        else:
            rsi_val = 50.0

        if self.USE_VOL:
            vol_avg = float(np.mean(volume[max(0, i - self.VOL_LEN): i])) \
                      if i > self.VOL_LEN else 1.0
            if float(volume[i]) < vol_avg * self.VOL_MULT:
                return []

        if self.USE_BASELINE:
            if np.isnan(smema_base[i]):
                return []
            if sig_type == "buy" and entry < float(smema_base[i]):
                return []
            if sig_type == "sell" and entry > float(smema_base[i]):
                return []

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl  = entry - self.ATR_SL_MULT  * atr_val
            tp1 = entry + self.TP1_ATR_MULT * atr_val
            tp2 = entry + self.TP2_ATR_MULT * atr_val
        else:
            sl  = entry + self.ATR_SL_MULT  * atr_val
            tp1 = entry - self.TP1_ATR_MULT * atr_val
            tp2 = entry - self.TP2_ATR_MULT * atr_val

        quality = self._quality(sig_type, rsi_val, context, htf_bias,
                                 smema_fast[i], smema_slow[i])

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp2, 5),      # engine targets full TP
            "quality":     quality,
            "zone":        {"high": round(float(smema_slow[i]) + atr_val, 5),
                            "low":  round(float(smema_slow[i]) - atr_val, 5)},
            "pattern_key": f"sovereign_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"SMEMA({self.FAST_LEN}/{self.SLOW_LEN}) cross "
                            f"{'up' if cross_up else 'down'} | "
                            f"TP1={tp1:.2f} (50%@{self.TP1_ATR_MULT}ATR) "
                            f"TP2={tp2:.2f} ({self.TP2_ATR_MULT}ATR) | "
                            f"exit=reversal_cross | "
                            f"max_bars={self.MAX_BARS} | "
                            f"RSI={rsi_val:.1f}"),
        }]

    # ── SMEMA: SMA( EMA(close, n), n ) ───────────────────────────────────────

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

    def _smema(self, arr: np.ndarray, n: int) -> np.ndarray:
        ema_arr = self._ema(arr, n)
        # SMA of ema_arr over n
        out = np.full(len(arr), np.nan)
        for k in range(n - 1, len(ema_arr)):
            window = ema_arr[k - n + 1: k + 1]
            if not np.any(np.isnan(window)):
                out[k] = window.mean()
        return out

    # ── ADX ───────────────────────────────────────────────────────────────────

    def _adx(self, high, low, close, i: int) -> float:
        n = self.ADX_LEN
        if i < n + 2:
            return 0.0
        sl  = slice(max(0, i - n - 2), i + 1)
        h   = high[sl];  l = low[sl];  c = close[sl]
        c1  = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        dm_p = np.where((h - np.roll(h, 1)) > (np.roll(l, 1) - l),
                         np.maximum(h - np.roll(h, 1), 0.0), 0.0)
        dm_m = np.where((np.roll(l, 1) - l) > (h - np.roll(h, 1)),
                         np.maximum(np.roll(l, 1) - l, 0.0), 0.0)

        atr14  = self._rma(tr,    n)[-1]
        di_p14 = 100 * self._rma(dm_p, n)[-1] / (atr14 + 1e-10)
        di_m14 = 100 * self._rma(dm_m, n)[-1] / (atr14 + 1e-10)
        dx     = 100 * abs(di_p14 - di_m14) / (di_p14 + di_m14 + 1e-10)
        return float(dx)

    @staticmethod
    def _rma(arr: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        if len(arr) < n:
            return out
        out[n - 1] = arr[:n].mean()
        alpha = 1.0 / n
        for k in range(n, len(arr)):
            out[k] = alpha * arr[k] + (1 - alpha) * out[k - 1]
        return out

    # ── RSI ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: np.ndarray, n: int) -> np.ndarray:
        out    = np.full(len(close), np.nan)
        delta  = np.diff(close, prepend=close[0])
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        if len(close) < n + 1:
            return out
        avg_g = gains[1:n + 1].mean()
        avg_l = losses[1:n + 1].mean()
        for k in range(n, len(close)):
            if k > n:
                avg_g = (avg_g * (n - 1) + gains[k]) / n
                avg_l = (avg_l * (n - 1) + losses[k]) / n
            rs     = avg_g / avg_l if avg_l > 0 else 100.0
            out[k] = 100 - 100 / (1 + rs)
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

    # ── Quality score ─────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, rsi: float, context: dict,
                 htf_bias: str, sf: float, ss: float) -> float:
        score = 5.5
        # SMEMA separation — wider gap = stronger trend
        sep = abs(sf - ss) / (abs(ss) + 1e-10)
        if sep > 0.005:
            score += 1.0
        # RSI in ideal range (not over-extended)
        if sig_type == "buy" and 50 <= rsi <= 65:
            score += 1.0
        elif sig_type == "sell" and 35 <= rsi <= 50:
            score += 1.0
        # ADX
        if context.get("adx", 0) > 30:
            score += 1.0
        # HTF bias alignment
        if sig_type == "buy" and htf_bias == "bullish":
            score += 0.5
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
