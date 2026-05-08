"""
strategies/asian_range_breakout.py
ADR Asian Range Breakout

Logic
-----
  Asian session range: scan the current df for rows where session column
  equals 'asian' (if the column is present), or fall back to the 20 bars
  with the smallest per-bar range (proxy for low-volatility / Asian hours).

  Range stored in a class-level _asian_cache dict keyed by df length to
  avoid recomputing on every tick.

  Breakout conditions:
    - Current session must be 'london' or 'new_york'
    - close breaks above asian_high → buy signal
    - close breaks below asian_low  → sell signal

  Range-size filter:
    - Asian range must be ≥ 0.3×ATR  (not trivially tight)
    - Asian range must be ≤ 3.0×ATR  (not excessively wide)

  SL  : buy → asian_low  − 0.5×ATR
        sell → asian_high + 0.5×ATR

  TP  : 1.5 × asian range size (ADR expansion target)
        trailing stop noted: trail by 0.5×ATR in notes.

  Quality (base 6):
    +1  range is 0.5–2×ATR  (ideal compression)
    +1  htf_bias aligned
    +1  session is 'london' (higher probability kill zone)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy


class AsianRangeBreakout(BaseStrategy):

    name        = "ADRAsianRangeBreakout"
    description = "Asian session range breakout — London / NY expansion"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    RANGE_MIN_ATR   = 0.3
    RANGE_MAX_ATR   = 3.0
    RANGE_IDEAL_LO  = 0.5
    RANGE_IDEAL_HI  = 2.0
    ATR_SL_BUFF     = 0.5
    TP_RANGE_MULT   = 1.5
    PROXY_BARS      = 20     # bars to use as Asian proxy when no session col
    MIN_BARS        = 25

    # Class-level cache: { cache_key: {'high': float, 'low': float} }
    _asian_cache: Dict[str, Dict] = {}

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

        # We only fire during London or New York sessions
        if session not in ("london", "new_york"):
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i       = len(df) - 1
        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else float(close[i]) * 0.001
        if atr_val <= 0:
            return []

        # ── Retrieve / build Asian range ──────────────────────────────────────
        asian_high, asian_low = self._get_asian_range(df, high, low, i)
        if asian_high is None or asian_low is None:
            return []

        asian_range = asian_high - asian_low
        if asian_range <= 0:
            return []

        # Range-size filter
        if asian_range < self.RANGE_MIN_ATR * atr_val:
            return []
        if asian_range > self.RANGE_MAX_ATR * atr_val:
            return []

        entry    = float(close[i])
        bull_brk = entry > asian_high
        bear_brk = entry < asian_low

        if not bull_brk and not bear_brk:
            return []

        sig_type = "buy" if bull_brk else "sell"

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl = asian_low  - self.ATR_SL_BUFF * atr_val
            tp = entry + self.TP_RANGE_MULT * asian_range
        else:
            sl = asian_high + self.ATR_SL_BUFF * atr_val
            tp = entry - self.TP_RANGE_MULT * asian_range

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        # ── Quality ───────────────────────────────────────────────────────────
        ideal_range = (
            self.RANGE_IDEAL_LO * atr_val
            <= asian_range
            <= self.RANGE_IDEAL_HI * atr_val
        )
        quality = self._score_quality(sig_type, ideal_range, session, htf_bias, context)

        notes = (
            f"Asian range [{round(asian_low,5)}–{round(asian_high,5)}] "
            f"({round(asian_range/atr_val,2)}×ATR) | "
            f"session={session} | "
            f"trail: trail SL by 0.5×ATR once in profit"
        )

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {
                "high": round(asian_high, 5),
                "low":  round(asian_low,  5),
            },
            "pattern_key": f"asian_range_brk_{sig_type}",
            "strategy":    self.name,
            "notes":       notes,
        }]

    # ── Asian range detection ─────────────────────────────────────────────────

    def _get_asian_range(
        self,
        df: pd.DataFrame,
        high: np.ndarray,
        low: np.ndarray,
        i: int,
    ):
        """
        Returns (asian_high, asian_low).
        Strategy:
          1. If df has a 'session' column, use rows marked 'asian' that
             precede the current bar.
          2. Otherwise use the PROXY_BARS bars ending at i-1 that have the
             smallest total bar ranges (low-volatility proxy for Asian session).
        """
        # ── Method 1: explicit session column ────────────────────────────────
        if "session" in df.columns:
            asian_mask = df["session"].values == "asian"
            # Exclude the current (potentially incomplete) bar
            asian_mask[i] = False
            if asian_mask.any():
                ah = float(np.max(high[asian_mask]))
                al = float(np.min(low[asian_mask]))
                return ah, al

        # ── Method 2: proxy — lowest-volatility bars ──────────────────────────
        lookback  = min(self.PROXY_BARS * 3, i)
        start     = i - lookback
        bar_range = high[start: i] - low[start: i]
        if len(bar_range) < self.PROXY_BARS:
            ah = float(np.max(high[start: i]))
            al = float(np.min(low[start:  i]))
            return ah, al

        # Pick PROXY_BARS bars with the smallest ranges
        sorted_idx = np.argsort(bar_range)[: self.PROXY_BARS]
        abs_idx    = sorted_idx + start
        ah = float(np.max(high[abs_idx]))
        al = float(np.min(low[abs_idx]))
        return ah, al

    # ── quality scoring ───────────────────────────────────────────────────────

    @staticmethod
    def _score_quality(
        sig_type: str,
        ideal_range: bool,
        session: str,
        htf_bias: str,
        context: Dict[str, Any],
    ) -> float:
        score = 6.0
        if ideal_range:
            score += 1.0
        bias = context.get("htf_bias", htf_bias)
        if sig_type == "buy"  and bias == "bullish":
            score += 1.0
        if sig_type == "sell" and bias == "bearish":
            score += 1.0
        if session == "london":
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)

    # ── shared helpers ────────────────────────────────────────────────────────

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
