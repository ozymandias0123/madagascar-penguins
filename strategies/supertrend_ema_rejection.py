"""
strategies/supertrend_ema_rejection.py
Supertrend + EMA Rejection

Logic
-----
  Three independent Supertrend bands:
    ST1: period=10, multiplier=2
    ST2: period=10, multiplier=3
    ST3: period=10, multiplier=5

  Entry — EMA-200 bounce / rejection:
    Buy:  price is above EMA200, all 3 STs are bullish (green),
          and the previous bar LOW touched/crossed ST1 (bounce off support)
    Sell: price is below EMA200, all 3 STs are bearish (red),
          and the previous bar HIGH touched/crossed ST1 (rejection)

  Exit triggers (encoded in notes; engine applies via TP/SL):
    - MACD histogram sign flip (managed externally)
    - Trailing: TP moves with EMA21 once in profit
    - Break-even: SL moves to entry after price moves 1×ATR in favour

  SL: beyond the widest Supertrend band (ST3) + small ATR buffer
  TP: 2× risk (R:R ≥ 2)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SupertrendEMAReject(BaseStrategy):

    name        = "SupertrendEMAReject"
    description = "3× Supertrend + EMA200 bounce/rejection entry"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    ST_PERIOD   = 10
    ST_MULTS    = (2.0, 3.0, 5.0)      # three independent multipliers
    EMA_TREND   = 200
    EMA_TRAIL   = 21
    RR_TARGET   = 2.0                   # minimum reward:risk
    ATR_SL_BUF  = 0.3                   # extra ATR buffer beyond ST3

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.EMA_TREND + 20:
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        # ── EMA 200 ───────────────────────────────────────────────────────────
        ema200 = self._ema(close, self.EMA_TREND)
        ema21  = self._ema(close, self.EMA_TRAIL)

        i = len(df) - 1
        if np.isnan(ema200[i]) or np.isnan(ema21[i]):
            return []

        # ── Three Supertrend bands ────────────────────────────────────────────
        st_vals  = []   # (upper_band, lower_band, direction)
        for mult in self.ST_MULTS:
            up, dn, direction = self._supertrend(high, low, close, atr_s,
                                                  self.ST_PERIOD, mult)
            st_vals.append((up, dn, direction))

        # Current direction: +1 = bullish, -1 = bearish
        dirs = [int(d[i]) for (_, _, d) in st_vals]
        all_bull = all(d == 1  for d in dirs)
        all_bear = all(d == -1 for d in dirs)

        if not all_bull and not all_bear:
            return []

        entry     = float(close[i])
        prev_low  = float(low[i - 1])
        prev_high = float(high[i - 1])
        atr_val   = float(atr_s[i]) if not np.isnan(atr_s[i]) else entry * 0.001

        # ST1 band value at current bar (support/resistance line)
        _, st1_dn, st1_dir = st_vals[0]
        _, st3_dn, _ = st_vals[2]
        st1_up, _, _ = st_vals[0]

        if all_bull:
            # Price must be above EMA200
            if entry <= ema200[i]:
                return []
            # Previous bar touched ST1 support band (bounce)
            st1_support = float(st1_dn[i - 1])
            if not (prev_low <= st1_support * 1.002):
                return []
            sig_type = "buy"
            # SL: below ST3 lower band + buffer
            sl = float(st3_dn[i]) - self.ATR_SL_BUF * atr_val

        else:  # all_bear
            # Price must be below EMA200
            if entry >= ema200[i]:
                return []
            # Previous bar touched ST1 resistance band (rejection)
            st1_resist = float(st1_up[i - 1])
            if not (prev_high >= st1_resist * 0.998):
                return []
            sig_type = "sell"
            # SL: above ST3 upper band + buffer
            st3_up, _, _ = st_vals[2]
            sl = float(st3_up[i]) + self.ATR_SL_BUF * atr_val

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = entry + risk * self.RR_TARGET if sig_type == "buy" \
             else entry - risk * self.RR_TARGET

        quality = self._quality(all_bull, context, entry, ema200[i], atr_val)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(float(high[i]), 5),
                            "low":  round(float(low[i]), 5)},
            "pattern_key": f"st_ema_reject_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"3×ST all {'bull' if all_bull else 'bear'} | "
                            f"EMA200 bounce | trail=EMA{self.EMA_TRAIL}"),
        }]

    # ── Supertrend ────────────────────────────────────────────────────────────

    @staticmethod
    def _supertrend(
        high:   np.ndarray,
        low:    np.ndarray,
        close:  np.ndarray,
        atr:    np.ndarray,
        period: int,
        mult:   float,
    ):
        """Returns (upper_band, lower_band, direction) arrays.
        direction: +1 = price above lower band (bullish),
                   -1 = price below upper band (bearish).
        """
        n = len(close)
        hl2       = (high + low) / 2.0
        upper_raw = hl2 + mult * atr
        lower_raw = hl2 - mult * atr

        upper = np.copy(upper_raw)
        lower = np.copy(lower_raw)
        dirn  = np.ones(n, dtype=int)

        for k in range(1, n):
            # lower band: ratchet up only
            lower[k] = lower_raw[k] if lower_raw[k] > lower[k - 1] \
                        or close[k - 1] < lower[k - 1] else lower[k - 1]
            # upper band: ratchet down only
            upper[k] = upper_raw[k] if upper_raw[k] < upper[k - 1] \
                        or close[k - 1] > upper[k - 1] else upper[k - 1]

            if dirn[k - 1] == -1 and close[k] > upper[k - 1]:
                dirn[k] = 1
            elif dirn[k - 1] == 1 and close[k] < lower[k - 1]:
                dirn[k] = -1
            else:
                dirn[k] = dirn[k - 1]

        return upper, lower, dirn

    # ── shared helpers ────────────────────────────────────────────────────────

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
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        c1 = np.roll(c, 1)
        c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    @staticmethod
    def _quality(bullish: bool, context: dict, price: float,
                 ema200: float, atr: float) -> float:
        score = 6.0
        dist_pct = abs(price - ema200) / (atr + 1e-10)
        # bonus when price is close to EMA200 (tight bounce)
        if dist_pct < 3:
            score += 1.0
        if context.get("adx", 0) > 25:
            score += 1.0
        if bullish and context.get("htf_bias", "") == "bullish":
            score += 1.0
        if not bullish and context.get("htf_bias", "") == "bearish":
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)
