"""
strategies/flowrise_bot.py
Flowrise BOT 2.0

Logic
-----
  Supertrend (period=10, multiplier=3) defines current trend direction.
  EMA200 acts as macro trend filter — longs only above, shorts only below.

  Session filter : London and New_York only.

  Entry trigger  : Supertrend flips direction on the current bar
                   (bearish→bullish for buy, bullish→bearish for sell)
                   AND close is on the correct side of EMA200.

  SL  : 1.5×ATR beyond entry price (below for buy, above for sell).
  TP  : 3.0×ATR from entry (trailing stop logic noted in notes field).

  Quality scoring (base 6):
    +1  htf_bias matches signal direction
    +1  ADX > 25 (trending market)
    +1  session is 'london' or 'new_york'
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy


class FlowriseBot(BaseStrategy):

    name        = "FlowriseBOT2"
    description = "Supertrend flip + EMA200 macro filter"
    version     = "2.0"

    # ── parameters ───────────────────────────────────────────────────────────
    ST_PERIOD   = 10
    ST_MULT     = 3.0
    EMA_PERIOD  = 200
    ATR_SL      = 1.5        # ATR multiples for stop-loss
    ATR_TP      = 3.0        # ATR multiples for take-profit
    MIN_BARS    = 220        # minimum bars needed

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

        # Session gate
        if session not in ("london", "new_york"):
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1

        # Current ATR value
        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else float(close[i]) * 0.001
        if atr_val <= 0:
            return []

        # ── EMA 200 ───────────────────────────────────────────────────────────
        ema200 = self._ema(close, self.EMA_PERIOD)
        if np.isnan(ema200[i]) or np.isnan(ema200[i - 1]):
            return []

        # ── Supertrend ────────────────────────────────────────────────────────
        _upper, _lower, direction = self._supertrend(
            high, low, close, atr_s, self.ST_PERIOD, self.ST_MULT
        )

        if i < 2:
            return []

        dir_now  = int(direction[i])
        dir_prev = int(direction[i - 1])

        # Detect flip
        flipped_bull = (dir_prev == -1) and (dir_now == 1)   # bearish → bullish
        flipped_bear = (dir_prev == 1)  and (dir_now == -1)  # bullish → bearish

        if not flipped_bull and not flipped_bear:
            return []

        entry = float(close[i])

        # EMA200 macro filter
        if flipped_bull and entry <= ema200[i]:
            return []
        if flipped_bear and entry >= ema200[i]:
            return []

        sig_type = "buy" if flipped_bull else "sell"

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl = entry - self.ATR_SL * atr_val
            tp = entry + self.ATR_TP * atr_val
        else:
            sl = entry + self.ATR_SL * atr_val
            tp = entry - self.ATR_TP * atr_val

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        # ── Quality ───────────────────────────────────────────────────────────
        quality = self._score_quality(sig_type, context, session)

        notes = (
            f"ST flip {'bull' if flipped_bull else 'bear'} | "
            f"EMA200={round(float(ema200[i]), 5)} | "
            f"session={session} | "
            f"trail: move SL to entry after 1×ATR in favour, then trail by 0.5×ATR"
        )

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {
                "high": round(float(high[i]), 5),
                "low":  round(float(low[i]),  5),
            },
            "pattern_key": f"flowrise_st_flip_{sig_type}",
            "strategy":    self.name,
            "notes":       notes,
        }]

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _supertrend(
        high:   np.ndarray,
        low:    np.ndarray,
        close:  np.ndarray,
        atr:    np.ndarray,
        period: int,
        mult:   float,
    ):
        """
        Returns (upper_band, lower_band, direction) arrays.
        direction : +1 = bullish (price above lower band)
                    -1 = bearish (price below upper band)
        """
        n         = len(close)
        hl2       = (high + low) / 2.0
        upper_raw = hl2 + mult * atr
        lower_raw = hl2 - mult * atr

        upper = np.copy(upper_raw)
        lower = np.copy(lower_raw)
        dirn  = np.ones(n, dtype=int)

        for k in range(1, n):
            lower[k] = (
                lower_raw[k]
                if (lower_raw[k] > lower[k - 1] or close[k - 1] < lower[k - 1])
                else lower[k - 1]
            )
            upper[k] = (
                upper_raw[k]
                if (upper_raw[k] < upper[k - 1] or close[k - 1] > upper[k - 1])
                else upper[k - 1]
            )

            if dirn[k - 1] == -1 and close[k] > upper[k - 1]:
                dirn[k] = 1
            elif dirn[k - 1] == 1 and close[k] < lower[k - 1]:
                dirn[k] = -1
            else:
                dirn[k] = dirn[k - 1]

        return upper, lower, dirn

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

    @staticmethod
    def _score_quality(sig_type: str, context: Dict[str, Any], session: str) -> float:
        score = 6.0
        bias  = context.get("htf_bias", context.get("bias", "neutral"))
        if sig_type == "buy"  and bias == "bullish":
            score += 1.0
        if sig_type == "sell" and bias == "bearish":
            score += 1.0
        if context.get("adx", 0) > 25:
            score += 1.0
        if session in ("london", "new_york"):
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)
