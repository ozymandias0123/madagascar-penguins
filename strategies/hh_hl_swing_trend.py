"""
strategies/hh_hl_swing_trend.py
HH/HL Swing Trend  (v1.0)

Logic
-----
  Pivot detection (pivotLookback = 10):
    pivot_high(k): high[k] == max(high[k-lb..k+lb])
    pivot_low(k):  low[k]  == min(low[k-lb..k+lb])

  Trend classification:
    uptrend   (trendDirection = 1):  lastHH > prevHH AND lastHL > prevHL
    downtrend (trendDirection = -1): lastLL < prevLL AND lastLH < prevLH

  Entry:
    Long:  trendDirection == 1
           AND low  <= lastHL + zoneHeightPips   (within zone above HL)
           AND low  >= lastHL - zoneHeightPips
           AND close > lastHL                    (closing above HL)
    Short: trendDirection == -1
           AND high >= lastLH - zoneHeightPips
           AND high <= lastLH + zoneHeightPips
           AND close < lastLH

  SL / TP:
    Long SL  = lastHL − zoneHeightPips
    Long TP  = lastHH + (lastHH − lastHL)       (project HH extension)
    Short SL = lastLH + zoneHeightPips
    Short TP = lastLL − (lastLH − lastLL)

  Trailing SL advisory encoded in notes:
    Long:  trail to new confirmed HL
    Short: trail to new confirmed LH

  zoneHeightPips = 5.0 (raw price distance, 1 pip = instrument-dependent)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from strategies.base_strategy import BaseStrategy


class HHHLSwingTrend(BaseStrategy):

    name        = "HHHLSwingTrend"
    description = "Pivot HH/HL uptrend, LH/LL downtrend; entry at swing retest zone"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    PIVOT_LB       = 10       # pivot lookback / confirmation window
    ZONE_PIPS      = 5.0      # half-height of entry zone around HL/LH
    MAX_SCAN       = 300      # max bars to scan back for pivots

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.PIVOT_LB * 2 + 5
        if len(df) < needed:
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values

        i = len(df) - 1

        # ── Collect pivot highs and lows ──────────────────────────────────────
        ph_vals, ph_bars = self._pivot_highs(high, i)
        pl_vals, pl_bars = self._pivot_lows(low,   i)

        if len(ph_vals) < 2 or len(pl_vals) < 2:
            return []

        # Most recent and previous swing high / low
        lastHH = ph_vals[0];  prevHH = ph_vals[1]
        lastHL = pl_vals[0];  prevHL = pl_vals[1]
        lastLH = ph_vals[0];  prevLH = ph_vals[1]   # same pivots, diff context
        lastLL = pl_vals[0];  prevLL = pl_vals[1]

        # ── Trend classification ──────────────────────────────────────────────
        uptrend   = lastHH > prevHH and lastHL > prevHL
        downtrend = lastLL < prevLL and lastLH < prevLH

        c_now = float(close[i])
        h_now = float(high[i])
        l_now = float(low[i])
        zone  = self.ZONE_PIPS

        long_cond  = (uptrend
                      and l_now <= lastHL + zone
                      and l_now >= lastHL - zone
                      and c_now  > lastHL)

        short_cond = (downtrend
                      and h_now >= lastLH - zone
                      and h_now <= lastLH + zone
                      and c_now  < lastLH)

        if not long_cond and not short_cond:
            return []

        # ── SL / TP ───────────────────────────────────────────────────────────
        if long_cond:
            sig_type = "buy"
            sl = lastHL - zone
            tp = lastHH + (lastHH - lastHL)
        else:
            sig_type = "sell"
            sl = lastLH + zone
            tp = lastLL - (lastLH - lastLL)

        entry = c_now
        risk  = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(sig_type, uptrend, downtrend, context, htf_bias)

        if long_cond:
            trail_note = f"trail_SL=new_HL | TP_ext={tp:.5f}"
        else:
            trail_note = f"trail_SL=new_LH | TP_ext={tp:.5f}"

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round((lastHL if long_cond else lastLH) + zone, 5),
                            "low":  round((lastHL if long_cond else lastLH) - zone, 5)},
            "pattern_key": f"hhhl_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"{'Up' if long_cond else 'Down'}trend | "
                            f"lastHL={lastHL:.5f} | lastHH={lastHH:.5f} | "
                            f"{trail_note}"),
        }]

    # ── Pivot helpers ─────────────────────────────────────────────────────────

    def _pivot_highs(self, high: np.ndarray, current: int
                     ) -> Tuple[List[float], List[int]]:
        lb = self.PIVOT_LB
        vals: List[float] = []
        bars: List[int]   = []
        window = lb * 2 + 1
        for k in range(current - lb, max(lb, current - self.MAX_SCAN), -1):
            if k - lb < 0 or k + lb >= len(high):
                continue
            seg = high[k - lb: k + lb + 1]
            if len(seg) == window and high[k] == np.max(seg):
                vals.append(float(high[k]))
                bars.append(k)
                if len(vals) >= 4:
                    break
        return vals, bars

    def _pivot_lows(self, low: np.ndarray, current: int
                    ) -> Tuple[List[float], List[int]]:
        lb = self.PIVOT_LB
        vals: List[float] = []
        bars: List[int]   = []
        window = lb * 2 + 1
        for k in range(current - lb, max(lb, current - self.MAX_SCAN), -1):
            if k - lb < 0 or k + lb >= len(low):
                continue
            seg = low[k - lb: k + lb + 1]
            if len(seg) == window and low[k] == np.min(seg):
                vals.append(float(low[k]))
                bars.append(k)
                if len(vals) >= 4:
                    break
        return vals, bars

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, uptrend: bool, downtrend: bool,
                 context: dict, htf_bias: str) -> float:
        score = 5.5
        if context.get("adx", 0) > 25:
            score += 1.0
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 1.0
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)
