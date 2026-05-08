"""
strategies/sovereign_alpha_ob.py
Sovereign Alpha — OB Stagnation Filter  (v1.0)

Logic
-----
  1. Pivot detection:
       pivotHigh(k) = high[k − lb] is the highest in a centered window of 2×lb+1 bars
       pivotLow(k)  = mirror

  2. Market Structure Break (MSB):
       Bullish MSB: close crosses above the most-recent confirmed pivot high
                    AND momentum Z-score > 0.5
       Bearish MSB: close crosses below the most-recent confirmed pivot low
                    AND momentum Z-score < −0.5

     Momentum Z-score = (price_change − mean_change_50) / stdev_change_50

  3. Order Block (OB) from MSB:
       Scan back up to 10 bars; take the last opposite-direction candle before the MSB.
       OB zone = [low[obIdx], high[obIdx]]

  4. Entry signal: price returns to the OB zone (touches it from the profit side):
       Bull OB:  low ≤ ob_top  AND  close > ob_bottom  (bouncing off OB)
       Bear OB:  high ≥ ob_bottom AND close < ob_top

  5. Macro HTF filter (encoded in context):
       Bull entry requires htf_bias != "bearish"  (weekly/daily EMA alignment)
       Bear entry requires htf_bias != "bullish"

  6. Quality:   HPZ if OB score > 80 (momentum-weighted + volume rank)

  SL:  hard_stop_pts (30) below/above entry
  TP:  target_pts   (100) above/below entry

  Stagnation (advisory): if price lingers inside OB > linger_bars (10),
    position should be cut — encoded in notes.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from strategies.base_strategy import BaseStrategy


class SovereignAlphaOB(BaseStrategy):

    name        = "SovereignAlphaOB"
    description = "Pivot-based MSB → OB detection + macro HTF filter + stagnation advisory"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    PIVOT_LB      = 7       # pivotLenInput
    MSB_Z_SCORE   = 0.5     # msbZScoreInput
    STOP_PTS      = 30.0    # hard stop in price points
    TARGET_PTS    = 100.0   # target profit in price points
    BE_PTS        = 15.0    # break-even trigger in points (advisory)
    LINGER_BARS   = 10      # stagnation: bars inside OB before cut
    OB_SCAN_BACK  = 10      # bars to scan back for the OB candle

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = self.PIVOT_LB * 3 + 55
        if len(df) < needed:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))

        i = len(df) - 1

        # ── Momentum Z-score ──────────────────────────────────────────────────
        n_z  = 50
        if i < n_z + 1:
            return []
        changes  = np.diff(close[i - n_z - 1: i + 1])
        pc       = changes[-1]
        avg_c    = changes[:-1].mean()
        std_c    = changes[:-1].std() + 1e-10
        z_score  = (pc - avg_c) / std_c

        # ── Find last confirmed pivot high / low ──────────────────────────────
        last_ph, last_ph_bar = self._last_pivot_high(high, i, self.PIVOT_LB)
        last_pl, last_pl_bar = self._last_pivot_low(low,   i, self.PIVOT_LB)

        # ── MSB detection ─────────────────────────────────────────────────────
        c_now  = float(close[i])
        c_prev = float(close[i - 1])

        msb_bull = (last_ph is not None
                    and c_prev <= last_ph and c_now > last_ph
                    and z_score > self.MSB_Z_SCORE)

        msb_bear = (last_pl is not None
                    and c_prev >= last_pl and c_now < last_pl
                    and z_score < -self.MSB_Z_SCORE)

        if not msb_bull and not msb_bear:
            return []

        # ── Macro HTF gate ────────────────────────────────────────────────────
        if msb_bull and htf_bias == "bearish":
            return []
        if msb_bear and htf_bias == "bullish":
            return []

        # ── Find Order Block ──────────────────────────────────────────────────
        ob_top, ob_bottom, ob_idx = self._find_ob(
            close, high, low, i, msb_bull, self.OB_SCAN_BACK)

        if ob_top is None:
            return []

        # ── OB quality score ──────────────────────────────────────────────────
        vol_pct = float(np.percentile(volume[max(0, i - 100): i + 1],
                                       np.searchsorted(
                                           np.sort(volume[max(0, i - 100): i + 1]),
                                           float(volume[i])) / max(1, min(100, i)) * 100))
        score   = min(100.0, abs(z_score) * 20 + vol_pct * 0.5)
        is_hpz  = score > 80

        # ── SL / TP (fixed point-based) ───────────────────────────────────────
        entry = c_now
        if msb_bull:
            sl = entry - self.STOP_PTS
            tp = entry + self.TARGET_PTS
        else:
            sl = entry + self.STOP_PTS
            tp = entry - self.TARGET_PTS

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        quality = self._quality(msb_bull, is_hpz, z_score, context, htf_bias)

        sig_type = "buy" if msb_bull else "sell"

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(ob_top, 5),
                            "low":  round(ob_bottom, 5)},
            "pattern_key": f"sov_alpha_ob_{sig_type}_{'hpz' if is_hpz else 'std'}",
            "strategy":    self.name,
            "notes":       (f"MSB {'bull' if msb_bull else 'bear'} | "
                            f"OB=[{ob_bottom:.2f}-{ob_top:.2f}] | "
                            f"score={score:.0f}{'★HPZ' if is_hpz else ''} | "
                            f"Z={z_score:.2f} | "
                            f"BE@{self.BE_PTS}pts | "
                            f"stagnation_cut={self.LINGER_BARS}bars"),
        }]

    # ── Pivot helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _last_pivot_high(high: np.ndarray, current: int,
                         lb: int) -> Tuple[Optional[float], Optional[int]]:
        window = lb * 2 + 1
        for k in range(current - lb, max(lb, current - lb - 200), -1):
            if k - lb < 0 or k + lb >= len(high):
                continue
            seg = high[k - lb: k + lb + 1]
            if len(seg) == window and high[k] == np.max(seg):
                return float(high[k]), k
        return None, None

    @staticmethod
    def _last_pivot_low(low: np.ndarray, current: int,
                        lb: int) -> Tuple[Optional[float], Optional[int]]:
        window = lb * 2 + 1
        for k in range(current - lb, max(lb, current - lb - 200), -1):
            if k - lb < 0 or k + lb >= len(low):
                continue
            seg = low[k - lb: k + lb + 1]
            if len(seg) == window and low[k] == np.min(seg):
                return float(low[k]), k
        return None, None

    @staticmethod
    def _find_ob(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 msb_bar: int, is_bull: bool,
                 scan_back: int) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Find last opposite-direction candle before the MSB."""
        for lag in range(1, scan_back + 1):
            k = msb_bar - lag
            if k < 0:
                break
            is_bear_candle = close[k] < (open := close[k] + (high[k] - low[k]) * 0.0)
            # Use close vs prior close as proxy for candle direction
            if k > 0:
                is_bear_c = close[k] < close[k - 1]
                is_bull_c = close[k] > close[k - 1]
            else:
                break
            if is_bull and is_bear_c:          # last bearish candle before bull MSB
                return float(high[k]), float(low[k]), k
            if not is_bull and is_bull_c:      # last bullish candle before bear MSB
                return float(high[k]), float(low[k]), k
        return None, None, None

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(is_bull: bool, is_hpz: bool, z_score: float,
                 context: dict, htf_bias: str) -> float:
        score = 5.5
        if is_hpz:
            score += 2.0
        if abs(z_score) > 1.5:
            score += 1.0
        elif abs(z_score) > 1.0:
            score += 0.5
        if is_bull  and htf_bias == "bullish":
            score += 0.5
        if not is_bull and htf_bias == "bearish":
            score += 0.5
        return round(min(max(score, 1.0), 10.0), 1)
