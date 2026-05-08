"""
strategies/prism_strategy.py
Prism Strategy (faithful Python port)

Logic (matches PineScript exactly)
------------------------------------
  MA stack:  SMA(5) > EMA(15) > EMA(30) > EMA(60)   → bullish
             SMA(5) < EMA(15) < EMA(30) < EMA(60)   → bearish

  RSI filter: RSI(14) > 50 for longs  /  < 50 for shorts

  Signal fires on the FIRST bar that transitions from no-stack to stack
  (trade_state flips from 0 → 1 or -1).  Re-arms after the stack fully
  reverses (trade_state back to 0, which happens when SL or TP is hit).

  SL:  low  − ATR × 1.5   (at signal bar)
  TP:  close + |close − SL| × 2.0   (R:R = 2)

  Structural trailing stop (advisory, encoded in notes):
    long:  max( EMA(12) − 0.2×ATR,  nearest_fib_below )
    short: min( EMA(12) + 0.2×ATR,  nearest_fib_above )

  Fibonacci levels are computed from highest/lowest over the last
  fib_lookback (100) bars; resistance levels above price and support
  levels below price are tracked the same way the Pine code does.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class PrismStrategy(BaseStrategy):

    name        = "PrismStrategy"
    description = "SMA5 > EMA15 > EMA30 > EMA60 stack + RSI + structural SL/TP"
    version     = "2.0"

    # ── parameters ───────────────────────────────────────────────────────────
    SMA_LEN     = 5        # sma1 in Pine
    EMA_MID     = 15       # ema15
    EMA_SLOW    = 30       # ema30
    EMA_BASE    = 60       # ema60
    EMA_WALL    = 12       # ema12 — structural wall reference
    RSI_LEN     = 14
    ATR_LEN     = 14
    ATR_MULT    = 1.5      # SL distance multiplier
    RR_RATIO    = 2.0      # TP = risk × RR
    FIB_LB      = 100      # fib lookback bars

    # Fibonacci retracement ratios (from swing high)
    _FIB_RATIOS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0, 1.272, 1.618]

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.EMA_BASE + self.FIB_LB + 5
        if len(df) < needed:
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        # ── MAs ───────────────────────────────────────────────────────────────
        sma5  = self._sma(close, self.SMA_LEN)
        ema15 = self._ema(close, self.EMA_MID)
        ema30 = self._ema(close, self.EMA_SLOW)
        ema60 = self._ema(close, self.EMA_BASE)
        ema12 = self._ema(close, self.EMA_WALL)
        rsi   = self._rsi(close, self.RSI_LEN)

        i = len(df) - 1

        if any(np.isnan(v) for v in [sma5[i], ema15[i], ema30[i], ema60[i],
                                      ema12[i], rsi[i], atr_s[i]]):
            return []

        # ── Stack detection (current bar) ─────────────────────────────────────
        bull_now = (sma5[i] > ema15[i] > ema30[i] > ema60[i]) and rsi[i] > 50
        bear_now = (sma5[i] < ema15[i] < ema30[i] < ema60[i]) and rsi[i] < 50

        # ── Check previous bar — signal only fires on fresh alignment ──────────
        prev = i - 1
        if not all(not np.isnan(v) for v in [sma5[prev], ema15[prev],
                                              ema30[prev], ema60[prev]]):
            return []

        bull_prev = (sma5[prev] > ema15[prev] > ema30[prev] > ema60[prev])
        bear_prev = (sma5[prev] < ema15[prev] < ema30[prev] < ema60[prev])

        if bull_now and bull_prev:
            return []   # stack was already bullish → not a new signal
        if bear_now and bear_prev:
            return []

        if not bull_now and not bear_now:
            return []

        sig_type = "buy" if bull_now else "sell"
        entry    = float(close[i])
        atr_val  = float(atr_s[i])

        # ── SL & TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl   = float(low[i]) - self.ATR_MULT * atr_val
        else:
            sl   = float(high[i]) + self.ATR_MULT * atr_val

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = entry + risk * self.RR_RATIO if sig_type == "buy" \
             else entry - risk * self.RR_RATIO

        # ── Fibonacci levels (for trailing stop note) ─────────────────────────
        lb       = min(self.FIB_LB, i)
        fib_high = float(np.max(high[i - lb: i + 1]))
        fib_low  = float(np.min(low[i - lb: i + 1]))
        fib_rng  = fib_high - fib_low

        fib_levels = sorted([fib_high - fib_rng * r for r in self._FIB_RATIOS])

        # Nearest fib below entry (support) and above entry (resistance)
        fib_sup = max((f for f in fib_levels if f < entry), default=fib_low)
        fib_res = min((f for f in fib_levels if f > entry), default=fib_high)

        # Structural wall trailing stop (advisory)
        e12 = float(ema12[i])
        if sig_type == "buy":
            struct_wall = max(e12 - 0.2 * atr_val, fib_sup) if e12 < entry \
                          else fib_sup
        else:
            struct_wall = min(e12 + 0.2 * atr_val, fib_res) if e12 > entry \
                          else fib_res

        quality = self._quality(bull_now, rsi[i], context)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(fib_res, 5), "low": round(fib_sup, 5)},
            "pattern_key": f"prism_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"MA stack {'bullish' if bull_now else 'bearish'} | "
                            f"RSI={rsi[i]:.1f} | "
                            f"struct_trail={struct_wall:.2f} | "
                            f"fib_sup={fib_sup:.2f} fib_res={fib_res:.2f}"),
        }]

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sma(arr: np.ndarray, n: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for k in range(n - 1, len(arr)):
            out[k] = arr[k - n + 1: k + 1].mean()
        return out

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

    @staticmethod
    def _rsi(arr: np.ndarray, n: int) -> np.ndarray:
        out    = np.full(len(arr), np.nan)
        delta  = np.diff(arr, prepend=arr[0])
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        if len(arr) < n + 1:
            return out
        avg_g = gains[1:n + 1].mean()
        avg_l = losses[1:n + 1].mean()
        for k in range(n, len(arr)):
            if k > n:
                avg_g = (avg_g * (n - 1) + gains[k]) / n
                avg_l = (avg_l * (n - 1) + losses[k]) / n
            rs = avg_g / avg_l if avg_l > 0 else 100.0
            out[k] = 100 - 100 / (1 + rs)
        return out

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

    @staticmethod
    def _quality(bullish: bool, rsi: float, context: dict) -> float:
        score = 5.5
        if bullish and rsi > 60:
            score += 1.5
        elif not bullish and rsi < 40:
            score += 1.5
        if context.get("adx", 0) > 25:
            score += 1.0
        if context.get("volatility", "normal") == "high":
            score -= 0.5
        return round(min(max(score, 1.0), 10.0), 1)
