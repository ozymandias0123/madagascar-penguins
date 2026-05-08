"""
strategies/pivot_vwap.py
Pivot + VWAP Strategy  (v2.0 — faithful Pine port)

Logic
-----
  Classic floor-trader pivot points (calculated from PREVIOUS day's H/L/C):
    P   = (H + L + C) / 3
    R1  = 2P − L,  R2 = P + (H−L),  R3 = R2 + (H−L)
    R4  = R3 + (H−L),               R5 = R4 + (H−L)
    S1  = 2P − H,  S2 = P − (H−L),  S3 = S2 − (H−L)
    S4  = S3 − (H−L),               S5 = S4 − (H−L)

  VWAP: session-resetting cumulative (HLC3 × volume) / cumulative volume.
  Resets when the calendar date changes.

  Entry rules (matches Pine check_long / check_short exactly):
    Buy:  low < pivot  AND  low < vwap  AND  close > pivot  AND  close > vwap
    Sell: high > pivot AND  high > vwap AND  close < pivot  AND  close < vwap

  All 11 levels (P, S1-S5, R1-R5) are tested each bar; the first matching
  level in order of proximity to price fires the signal.

  Exit (advisory, encoded in notes):
    Primary: 2 consecutive closes back below/above VWAP
    TP anchor: next pivot level in the profit direction (used for RR calc)
    SL: ATR buffer beyond the triggering pivot level

  Time filter: 09:30 – 14:00 (exchange local time).
  Max 2 signals per calendar day.
"""

import numpy as np
import pandas as pd
from datetime import time as dtime
from typing import Any, Dict, List, Optional, Tuple
from strategies.base_strategy import BaseStrategy


class PivotVWAP(BaseStrategy):

    name        = "PivotVWAP"
    description = "Daily pivots + VWAP wick-bounce entry, 09:30-14:00, max 2/day"
    version     = "2.0"

    # ── parameters ───────────────────────────────────────────────────────────
    SESSION_START = dtime(9, 30)
    SESSION_END   = dtime(14, 0)
    MAX_DAILY     = 2
    ATR_SL_MULT   = 0.5      # SL buffer beyond triggering pivot
    MIN_RR        = 1.0

    def __init__(self):
        super().__init__()
        self._daily_count: Dict[str, int] = {}   # date_str → count

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < 50:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns \
                 else np.ones(len(df))
        atr_s  = df["atr"].values if "atr" in df.columns \
                 else self._calc_atr_arr(df)

        # ── Time filter ───────────────────────────────────────────────────────
        if not self._in_session(df):
            return []

        today_str = self._today_str(df)
        if self._daily_count.get(today_str, 0) >= self.MAX_DAILY:
            return []

        # ── Previous-day OHLC for pivots ──────────────────────────────────────
        prev_h, prev_l, prev_c = self._prev_day_hlc(df)
        if prev_h is None:
            return []

        pivots = self._calc_pivots(prev_h, prev_l, prev_c)

        # ── VWAP (session-reset) ──────────────────────────────────────────────
        vwap = self._calc_vwap(df, close, high, low, volume)

        i        = len(df) - 1
        c_now    = float(close[i])
        h_now    = float(high[i])
        l_now    = float(low[i])
        atr_val  = float(atr_s[i]) if not np.isnan(atr_s[i]) else c_now * 0.001
        vwap_now = float(vwap[i])

        if np.isnan(vwap_now):
            return []

        # ── Pine: check_long(p) ───────────────────────────────────────────────
        #   low < p AND low < vwap AND close > p AND close > vwap
        # ── Pine: check_short(p) ─────────────────────────────────────────────
        #   high > p AND high > vwap AND close < p AND close < vwap

        sig: Optional[Tuple] = None

        # For a buy:  wick dipped below pivot AND vwap, close above both.
        #   Iterate HIGH → LOW so we pick the tightest (closest-to-close) pivot.
        # For a sell: wick pushed above pivot AND vwap, close below both.
        #   Iterate LOW → HIGH for the same reason.
        all_named_asc  = pivots["all_named"]          # ascending by value
        all_named_desc = list(reversed(all_named_asc))

        for lvl_name, lvl_val in all_named_desc:
            # check_long: low < p AND low < vwap AND close > p AND close > vwap
            if (l_now < lvl_val and l_now < vwap_now
                    and c_now > lvl_val and c_now > vwap_now):
                sl  = lvl_val - self.ATR_SL_MULT * atr_val
                tp  = self._next_pivot_above(c_now, pivots, atr_val)
                rr  = self._rr(c_now, sl, tp)
                if rr >= self.MIN_RR:
                    sig = ("buy", c_now, sl, tp, lvl_name, rr)
                    break

        if sig is None:
            for lvl_name, lvl_val in all_named_asc:
                # check_short: high > p AND high > vwap AND close < p AND close < vwap
                if (h_now > lvl_val and h_now > vwap_now
                        and c_now < lvl_val and c_now < vwap_now):
                    sl  = lvl_val + self.ATR_SL_MULT * atr_val
                    tp  = self._next_pivot_below(c_now, pivots, atr_val)
                    rr  = self._rr(c_now, sl, tp)
                    if rr >= self.MIN_RR:
                        sig = ("sell", c_now, sl, tp, lvl_name, rr)
                        break

        if sig is None:
            return []

        sig_type, entry, sl, tp, pivot_name, rr = sig
        quality = self._quality(sig_type, rr, context, entry, vwap_now)

        self._daily_count[today_str] = self._daily_count.get(today_str, 0) + 1

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(pivots["P"] + atr_val, 5),
                            "low":  round(pivots["P"] - atr_val, 5)},
            "pattern_key": f"pivot_vwap_{sig_type}_{pivot_name}",
            "strategy":    self.name,
            "notes":       (f"Pivot {pivot_name} | VWAP={vwap_now:.2f} | "
                            f"RR={rr:.1f} | exit=2-close-VWAP | "
                            f"day#{self._daily_count[today_str]}"),
        }]

    # ── Pivot math ────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_pivots(prev_h: float, prev_l: float, prev_c: float) -> dict:
        P   = (prev_h + prev_l + prev_c) / 3.0
        rng = prev_h - prev_l

        R1 = 2 * P - prev_l;  R2 = P + rng;    R3 = R2 + rng
        R4 = R3 + rng;         R5 = R4 + rng

        S1 = 2 * P - prev_h;  S2 = P - rng;    S3 = S2 - rng
        S4 = S3 - rng;         S5 = S4 - rng

        named = [("P",  P),
                 ("R1", R1), ("R2", R2), ("R3", R3), ("R4", R4), ("R5", R5),
                 ("S1", S1), ("S2", S2), ("S3", S3), ("S4", S4), ("S5", S5)]

        all_vals  = sorted(v for _, v in named)
        all_named = sorted(named, key=lambda x: x[1])

        return dict(P=P, all_named=all_named, all_levels=all_vals)

    def _next_pivot_above(self, price: float, pivots: dict,
                          atr: float) -> float:
        for lvl in pivots["all_levels"]:
            if lvl > price + atr * 0.1:
                return lvl
        return price + atr * 2.0

    def _next_pivot_below(self, price: float, pivots: dict,
                          atr: float) -> float:
        for lvl in reversed(pivots["all_levels"]):
            if lvl < price - atr * 0.1:
                return lvl
        return price - atr * 2.0

    # ── VWAP ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_vwap(df: pd.DataFrame, close, high, low, volume) -> np.ndarray:
        """Session-resetting VWAP; resets when calendar date changes."""
        hlc3   = (high + low + close) / 3.0
        vwap   = np.full(len(close), np.nan)
        cum_pv = 0.0
        cum_v  = 0.0
        prev_date = None

        dates = None
        if hasattr(df.index, "date"):
            try:
                dates = [d.date() for d in df.index]
            except Exception:
                pass

        for k in range(len(close)):
            cur_date = dates[k] if dates is not None else None
            if cur_date != prev_date:
                cum_pv    = 0.0
                cum_v     = 0.0
                prev_date = cur_date
            v       = float(volume[k]) if volume[k] > 0 else 1.0
            cum_pv += float(hlc3[k]) * v
            cum_v  += v
            vwap[k] = cum_pv / cum_v

        return vwap

    # ── time / date helpers ───────────────────────────────────────────────────

    def _in_session(self, df: pd.DataFrame) -> bool:
        try:
            t = df.index[-1].time()
            return self.SESSION_START <= t <= self.SESSION_END
        except Exception:
            return True

    @staticmethod
    def _today_str(df: pd.DataFrame) -> str:
        try:
            return str(df.index[-1].date())
        except Exception:
            return "unknown"

    @staticmethod
    def _prev_day_hlc(df: pd.DataFrame):
        """Find previous day's H/L/C from intraday bars."""
        try:
            dates = [d.date() for d in df.index]
        except Exception:
            idx    = max(0, len(df) - 289)
            prev_h = float(df["high"].iloc[idx:-1].max())
            prev_l = float(df["low"].iloc[idx:-1].min())
            prev_c = float(df["close"].iloc[-2])
            return prev_h, prev_l, prev_c

        today     = dates[-1]
        prev_bars = [i for i, d in enumerate(dates) if d < today]
        if not prev_bars:
            return None, None, None

        prev_day      = dates[prev_bars[-1]]
        prev_day_bars = [i for i, d in enumerate(dates) if d == prev_day]

        prev_h = float(df["high"].iloc[prev_day_bars].max())
        prev_l = float(df["low"].iloc[prev_day_bars].min())
        prev_c = float(df["close"].iloc[prev_day_bars[-1]])
        return prev_h, prev_l, prev_c

    # ── quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, rr: float, context: dict,
                 price: float, vwap: float) -> float:
        score = 5.0
        score += min(rr - 1.0, 2.0)
        dist_pct = abs(price - vwap) / (vwap + 1e-10)
        if dist_pct < 0.003:
            score += 1.0
        if context.get("adx", 0) > 20:
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)

    # ── ATR fallback ─────────────────────────────────────────────────────────

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
