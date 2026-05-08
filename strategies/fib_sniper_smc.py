"""
strategies/fib_sniper_smc.py
FibSniper Pro SMC

Logic
-----
  Swing detection : highest high / lowest low over the last 20 bars.

  Fibonacci retracement levels computed from swing low→high (bull) or
  swing high→low (bear): 0.236, 0.382, 0.5, 0.618, 0.786.

  SMC Order Blocks
    Bullish OB : bearish candle immediately before a bullish impulse
                 (the 3-bar sequence: bar[-3] bearish, bar[-2] or bar[-1]
                 strongly bullish and range > 1×ATR).
    Bearish OB : bullish candle immediately before a bearish impulse.

  BOS (Break of Structure)
    Bullish BOS : current close > last 20-bar swing high.
    Bearish BOS : current close < last 20-bar swing low.

  Entry conditions (all required):
    - Price is in 0.382–0.618 Fibonacci zone
    - Price is near Order Block (within 0.5×ATR)
    - BOS confirmed in the trade direction
    - Engulfing candle on the current bar (body engulfs prior bar body)

  SL  : below OB low − 0.3×ATR (buy) | above OB high + 0.3×ATR (sell)
  TP  : entry ± 2.5 × risk

  Quality (base 6):
    +1  BOS confirmed
    +1  Engulfing pattern
    +1  htf_bias aligned
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy


class FibSniperSMC(BaseStrategy):

    name        = "FibSniperProSMC"
    description = "Fibonacci retracement + SMC order blocks + BOS"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    SWING_PERIOD  = 20
    FIB_ZONE_LO   = 0.382
    FIB_ZONE_HI   = 0.618
    OB_PROXIMITY  = 0.5     # ATR multiples to consider "near OB"
    SL_BUFFER     = 0.3     # ATR buffer beyond OB
    RR_TARGET     = 2.5
    MIN_BARS      = 30

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
        high  = df["high"].values
        low   = df["low"].values
        op    = df["open"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1

        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else float(close[i]) * 0.001
        if atr_val <= 0:
            return []

        # ── Swing high / low over last SWING_PERIOD bars ─────────────────────
        lookback    = min(self.SWING_PERIOD, i)
        start       = i - lookback
        swing_high  = float(np.max(high[start: i + 1]))
        swing_low   = float(np.min(low[start:  i + 1]))
        swing_range = swing_high - swing_low
        if swing_range < 1e-10:
            return []

        entry = float(close[i])

        # ── BOS detection ────────────────────────────────────────────────────
        bos_bull = entry > swing_high   # closed above recent swing high
        bos_bear = entry < swing_low    # closed below recent swing low

        # ── Fibonacci levels ─────────────────────────────────────────────────
        fibs = self._fib_levels(swing_low, swing_high)

        # ── Order Block detection ─────────────────────────────────────────────
        ob_bull = self._detect_ob_bull(op, close, high, low, i, atr_val)
        ob_bear = self._detect_ob_bear(op, close, high, low, i, atr_val)

        # ── Engulfing check ───────────────────────────────────────────────────
        engulf_bull = self._engulfing_bull(op, close, i)
        engulf_bear = self._engulfing_bear(op, close, i)

        signals = []

        # ── BUY setup ─────────────────────────────────────────────────────────
        if bos_bull and ob_bull is not None:
            in_fib_zone = fibs["0.618"] <= entry <= fibs["0.382_inv"]
            # For a bullish move: fib zone is between swing_low + 0.382×range
            #                     and swing_low + 0.618×range
            fib_lo = swing_low + self.FIB_ZONE_LO * swing_range
            fib_hi = swing_low + self.FIB_ZONE_HI * swing_range
            in_fib_zone = (fib_lo <= entry <= fib_hi)
            near_ob     = abs(entry - ob_bull["mid"]) <= self.OB_PROXIMITY * atr_val

            if in_fib_zone and near_ob and engulf_bull:
                sl = ob_bull["low"] - self.SL_BUFFER * atr_val
                risk = abs(entry - sl)
                if risk > 1e-10:
                    tp      = entry + self.RR_TARGET * risk
                    quality = self._score(True, True, engulf_bull, htf_bias, context)
                    signals.append({
                        "type":        "buy",
                        "entry_price": round(entry, 5),
                        "sl_price":    round(sl, 5),
                        "tp_price":    round(tp, 5),
                        "quality":     quality,
                        "zone":        {
                            "high": round(fib_hi, 5),
                            "low":  round(fib_lo, 5),
                        },
                        "pattern_key": "fib_smc_bos_buy",
                        "strategy":    self.name,
                        "notes": (
                            f"BOS bull | OB bull [{round(ob_bull['low'],5)}"
                            f"–{round(ob_bull['high'],5)}] | "
                            f"Fib zone [{round(fib_lo,5)}–{round(fib_hi,5)}] | "
                            f"Engulf={engulf_bull}"
                        ),
                    })

        # ── SELL setup ────────────────────────────────────────────────────────
        if bos_bear and ob_bear is not None:
            fib_lo = swing_high - self.FIB_ZONE_HI * swing_range
            fib_hi = swing_high - self.FIB_ZONE_LO * swing_range
            in_fib_zone = (fib_lo <= entry <= fib_hi)
            near_ob     = abs(entry - ob_bear["mid"]) <= self.OB_PROXIMITY * atr_val

            if in_fib_zone and near_ob and engulf_bear:
                sl = ob_bear["high"] + self.SL_BUFFER * atr_val
                risk = abs(entry - sl)
                if risk > 1e-10:
                    tp      = entry - self.RR_TARGET * risk
                    quality = self._score(True, True, engulf_bear, htf_bias, context)
                    signals.append({
                        "type":        "sell",
                        "entry_price": round(entry, 5),
                        "sl_price":    round(sl, 5),
                        "tp_price":    round(tp, 5),
                        "quality":     quality,
                        "zone":        {
                            "high": round(fib_hi, 5),
                            "low":  round(fib_lo, 5),
                        },
                        "pattern_key": "fib_smc_bos_sell",
                        "strategy":    self.name,
                        "notes": (
                            f"BOS bear | OB bear [{round(ob_bear['low'],5)}"
                            f"–{round(ob_bear['high'],5)}] | "
                            f"Fib zone [{round(fib_lo,5)}–{round(fib_hi,5)}] | "
                            f"Engulf={engulf_bear}"
                        ),
                    })

        return signals

    # ── SMC helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_ob_bull(op, close, high, low, i, atr_val):
        """Bullish OB: bearish candle before a bullish impulse."""
        if i < 3:
            return None
        # bar[-3]: bearish candle
        if close[i - 3] >= op[i - 3]:
            return None
        # bar[-2] or current bar: bullish impulse > 0.5×ATR
        impulse = close[i - 2] - op[i - 2]
        if impulse < 0.5 * atr_val:
            return None
        ob_high = float(max(op[i - 3], close[i - 3]))
        ob_low  = float(min(op[i - 3], close[i - 3]))
        return {"high": ob_high, "low": ob_low, "mid": (ob_high + ob_low) / 2}

    @staticmethod
    def _detect_ob_bear(op, close, high, low, i, atr_val):
        """Bearish OB: bullish candle before a bearish impulse."""
        if i < 3:
            return None
        if close[i - 3] <= op[i - 3]:
            return None
        impulse = op[i - 2] - close[i - 2]   # negative impulse
        if impulse < 0.5 * atr_val:
            return None
        ob_high = float(max(op[i - 3], close[i - 3]))
        ob_low  = float(min(op[i - 3], close[i - 3]))
        return {"high": ob_high, "low": ob_low, "mid": (ob_high + ob_low) / 2}

    @staticmethod
    def _engulfing_bull(op, close, i) -> bool:
        if i < 1:
            return False
        prev_body = abs(close[i - 1] - op[i - 1])
        curr_body = close[i] - op[i]
        return (curr_body > 0) and (curr_body > prev_body) and (close[i - 1] < op[i - 1])

    @staticmethod
    def _engulfing_bear(op, close, i) -> bool:
        if i < 1:
            return False
        prev_body = abs(close[i - 1] - op[i - 1])
        curr_body = op[i] - close[i]
        return (curr_body > 0) and (curr_body > prev_body) and (close[i - 1] > op[i - 1])

    @staticmethod
    def _fib_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
        r = swing_high - swing_low
        return {
            "0.236":     swing_high - 0.236 * r,
            "0.382":     swing_high - 0.382 * r,
            "0.5":       swing_high - 0.500 * r,
            "0.618":     swing_high - 0.618 * r,
            "0.786":     swing_high - 0.786 * r,
            "0.382_inv": swing_low  + 0.382 * r,
        }

    @staticmethod
    def _score(bos: bool, near_ob: bool, engulf: bool,
               htf_bias: str, context: Dict[str, Any]) -> float:
        score = 6.0
        if bos:
            score += 1.0
        if engulf:
            score += 1.0
        bias = context.get("htf_bias", htf_bias)
        if bias in ("bullish", "bearish"):
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)

    # ── shared low-level helpers ──────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        out  = np.full(len(arr), np.nan)
        mult = 2.0 / (n + 1)
        if len(arr) < n:
            return out
        out[n - 1] = arr[:n].mean()
        for k in range(n, len(arr)):
            out[k] = arr[k] * mult + out[k - 1] * (1.0 - mult)
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
