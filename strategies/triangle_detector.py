"""
strategies/triangle_detector.py
Triangle Pattern Detector

Scans for Symmetrical, Ascending, Descending triangles
and Wedges using linear regression slopes on highs/lows.
Generates signal on confirmed breakout.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from strategies.base_strategy import BaseStrategy


class TriangleDetector(BaseStrategy):

    name        = "TriangleDetector"
    description = "Triangle/Wedge pattern + breakout confirmation"
    version     = "1.0"

    WINDOW      = 30        # bars to scan for pattern
    MIN_POINTS  = 5
    TOLERANCE   = 0.015     # slope flatness threshold
    BREAKOUT_PCT = 0.005    # 0.5% beyond pattern boundary = confirmed

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.WINDOW + self.MIN_POINTS + 5:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        atr_s  = df["atr"] if "atr" in df.columns else self._calc_atr(df, 14)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        # scan the last WINDOW bars ending at i
        end   = len(df) + i + 1          # exclusive
        start = max(0, end - self.WINDOW)
        win_h = high.values[start:end]
        win_l = low.values[start:end]

        if len(win_h) < self.MIN_POINTS:
            return []

        x         = np.arange(len(win_h))
        high_slope = float(np.polyfit(x, win_h, 1)[0])
        low_slope  = float(np.polyfit(x, win_l, 1)[0])

        # normalise slopes by price magnitude
        norm       = entry if entry != 0 else 1
        hs         = high_slope / norm
        ls         = low_slope  / norm
        tol        = self.TOLERANCE

        pattern = self._classify(hs, ls, tol)
        if pattern is None:
            return []

        pat_high = float(win_h.max())
        pat_low  = float(win_l.min())

        direction = self._breakout_direction(entry, pat_high, pat_low)
        if direction == "Forming":
            return []                    # no signal until breakout

        signals = []
        strength = abs(hs) + abs(ls)

        if direction == "BULLISH BREAKOUT":
            sl = pat_low - atr_v * 0.5
            tp = entry + (entry - sl) * 2.0
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     self._quality(pattern, strength),
                "zone":        {"high": pat_high, "low": pat_low},
                "pattern_key": f"triangle_{pattern.lower().replace(' ', '_')}_bull",
                "strategy":    self.name,
                "notes":       f"{pattern} bullish breakout, hs={hs:.5f} ls={ls:.5f}",
            })

        elif direction == "BEARISH BREAKOUT":
            sl = pat_high + atr_v * 0.5
            tp = entry - (sl - entry) * 2.0
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     self._quality(pattern, strength),
                "zone":        {"high": pat_high, "low": pat_low},
                "pattern_key": f"triangle_{pattern.lower().replace(' ', '_')}_bear",
                "strategy":    self.name,
                "notes":       f"{pattern} bearish breakout, hs={hs:.5f} ls={ls:.5f}",
            })

        return signals

    # ── helpers ──────────────────────────────────────────────

    def _classify(self, hs: float, ls: float, tol: float) -> Optional[str]:
        if abs(hs) < tol and abs(ls) < tol:
            return "Symmetrical Triangle"
        if hs < -tol and ls > tol:
            return "Ascending Triangle" if ls > abs(hs) else "Ascending Wedge"
        if hs > tol and ls < -tol:
            return "Descending Triangle" if hs > abs(ls) else "Descending Wedge"
        return None

    def _breakout_direction(self, price: float, pat_high: float, pat_low: float) -> str:
        if price > pat_high * (1 - self.BREAKOUT_PCT):
            return "BULLISH BREAKOUT"
        if price < pat_low  * (1 + self.BREAKOUT_PCT):
            return "BEARISH BREAKOUT"
        return "Forming"

    def _quality(self, pattern: str, strength: float) -> float:
        base = {
            "Ascending Triangle":  7.5,
            "Descending Triangle": 7.5,
            "Symmetrical Triangle": 6.5,
            "Ascending Wedge":     6.0,
            "Descending Wedge":    6.0,
        }.get(pattern, 5.0)
        return min(base + strength * 10, 10.0)

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
