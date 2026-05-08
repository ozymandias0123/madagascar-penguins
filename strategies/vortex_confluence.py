"""
strategies/vortex_confluence.py
Vortex Confluence Protocol

Logic
-----
  Multi-layer scoring system (0-100 points). Fire signal when score >= 65.

  Layer 1 — Trend (0-25 pts):
    HTF bias match   : +10
    Price vs EMA200  : +8  (above for buy, below for sell)
    ADX > 25         : +7

  Layer 2 — Momentum (0-25 pts):
    RSI ideal range  : +10  (45-65 buy / 35-55 sell)
    MACD hist dir    : +8
    RSI trending     : +7   (RSI[i] > RSI[i-2] for buy)

  Layer 3 — Structure/SMC (0-25 pts):
    BOS in direction : +10
    FVG / OB zone    : +8   (3-bar fair-value-gap)
    Liquidity sweep  : +7   (wick rejection of recent swing)

  Layer 4 — Entry timing (0-25 pts):
    Kill-zone session: +10  (london / new_york)
    Candle pattern   : +8   (engulfing or pin bar)
    ATR expansion    : +7   (current ATR > 20-bar avg ATR)

  TP multiplier: score >= 80 → 3R, otherwise 2R
  Quality      : score / 10 (capped 1-10)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class VortexConfluence(BaseStrategy):

    name        = "VortexConfluence"
    description = "4-layer 100-point scoring: trend + momentum + SMC + timing"
    version     = "1.0"

    MIN_SCORE  = 65
    EMA_LEN    = 200
    SWING_LOOK = 20

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.EMA_LEN + 10:
            return []

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        open_  = df["open"].values
        atr_s  = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)
        rsi_s  = df["rsi"].values if "rsi" in df.columns else self._calc_rsi(close)

        i = len(df) - 1
        if np.isnan(atr_s[i]) or np.isnan(rsi_s[i]):
            return []

        entry   = float(close[i])
        atr_val = float(atr_s[i])

        ema200 = self._ema(close, self.EMA_LEN)
        if np.isnan(ema200[i]):
            return []

        best = None
        best_score = 0

        for sig_type in ("buy", "sell"):
            score = self._score_all(
                sig_type, i, close, high, low, open_,
                atr_s, rsi_s, ema200, context, session, htf_bias, atr_val, entry,
            )
            if score >= self.MIN_SCORE and score > best_score:
                best_score = score
                sl   = entry - 1.5 * atr_val if sig_type == "buy" else entry + 1.5 * atr_val
                rr   = 3.0 if score >= 80 else 2.0
                risk = abs(entry - sl)
                tp   = (entry + risk * rr) if sig_type == "buy" else (entry - risk * rr)
                best = {
                    "type":        sig_type,
                    "entry_price": round(entry, 5),
                    "sl_price":    round(sl, 5),
                    "tp_price":    round(tp, 5),
                    "quality":     round(min(score / 10.0, 10.0), 1),
                    "zone":        {"high": round(float(high[i]), 5),
                                    "low":  round(float(low[i]), 5)},
                    "pattern_key": f"vortex_confluence_{sig_type}",
                    "strategy":    self.name,
                    "notes":       f"Score={score}/100 | RR={rr:.1f} | session={session}",
                }

        return [best] if best else []

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_all(self, sig_type, i, close, high, low, open_,
                   atr_s, rsi_s, ema200, context, session, htf_bias,
                   atr_val, entry) -> int:
        score  = 0
        is_buy = sig_type == "buy"

        # ---- Layer 1: Trend ------------------------------------------------
        if (is_buy and htf_bias == "bullish") or (not is_buy and htf_bias == "bearish"):
            score += 10
        if (is_buy and entry > ema200[i]) or (not is_buy and entry < ema200[i]):
            score += 8
        if context.get("adx", 0) > 25:
            score += 7

        # ---- Layer 2: Momentum ---------------------------------------------
        rsi = float(rsi_s[i])
        if is_buy  and 45.0 <= rsi <= 65.0: score += 10
        if not is_buy and 35.0 <= rsi <= 55.0: score += 10

        # MACD histogram from context (may not always be present)
        mh = context.get("macd_hist", None)
        if mh is not None:
            if (is_buy and mh > 0) or (not is_buy and mh < 0):
                score += 8

        if i >= 2:
            r_now  = float(rsi_s[i])
            r_prev = float(rsi_s[i - 2]) if not np.isnan(rsi_s[i - 2]) else r_now
            if (is_buy and r_now > r_prev) or (not is_buy and r_now < r_prev):
                score += 7

        # ---- Layer 3: Structure / SMC --------------------------------------
        look = min(self.SWING_LOOK, i - 1)
        if look > 1:
            swing_high = float(np.max(high[i - look: i]))
            swing_low  = float(np.min(low[i - look: i]))

            # BOS
            if is_buy  and entry > swing_high: score += 10
            if not is_buy and entry < swing_low: score += 10

            # FVG — 3-bar fair-value gap
            if i >= 3:
                if is_buy  and float(low[i - 1])  > float(high[i - 3]): score += 8
                if not is_buy and float(high[i - 1]) < float(low[i - 3]):  score += 8

            # Liquidity sweep: wick beyond swing then price reclaimed
            if is_buy  and float(low[i])  < swing_low  and entry > swing_low:  score += 7
            if not is_buy and float(high[i]) > swing_high and entry < swing_high: score += 7

        # ---- Layer 4: Timing -----------------------------------------------
        if session in ("london", "new_york"):
            score += 10

        # Candle pattern — engulfing or pin bar
        if i >= 1:
            prev_body  = abs(float(close[i - 1]) - float(open_[i - 1]))
            curr_body  = abs(float(close[i])     - float(open_[i]))
            total_rng  = float(high[i]) - float(low[i])
            if curr_body > prev_body * 1.1:
                # Engulfing
                if (is_buy  and float(close[i]) > float(open_[i])) or \
                   (not is_buy and float(close[i]) < float(open_[i])):
                    score += 8
            elif total_rng > 0 and curr_body / total_rng < 0.35:
                # Pin bar
                lower_wick = min(float(open_[i]), float(close[i])) - float(low[i])
                upper_wick = float(high[i]) - max(float(open_[i]), float(close[i]))
                if is_buy  and lower_wick > total_rng * 0.5: score += 8
                if not is_buy and upper_wick > total_rng * 0.5: score += 8

        # ATR expansion
        if i >= 20:
            avg_atr = float(np.nanmean(atr_s[i - 20: i]))
            if avg_atr > 0 and atr_val > avg_atr * 1.1:
                score += 7

        return score

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ema(arr: np.ndarray, n: int) -> np.ndarray:
        out  = np.full(len(arr), np.nan)
        if len(arr) < n:
            return out
        mult = 2.0 / (n + 1)
        out[n - 1] = arr[:n].mean()
        for k in range(n, len(arr)):
            out[k] = arr[k] * mult + out[k - 1] * (1 - mult)
        return out

    @staticmethod
    def _calc_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
        out = np.full(len(close), np.nan)
        if len(close) < n + 1:
            return out
        delta = np.diff(close)
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = gain[:n].mean()
        avg_l = loss[:n].mean()
        for k in range(n, len(delta)):
            avg_g = (avg_g * (n - 1) + gain[k]) / n
            avg_l = (avg_l * (n - 1) + loss[k]) / n
            rs    = avg_g / (avg_l + 1e-10)
            out[k + 1] = 100 - 100 / (1 + rs)
        return out

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values
        l  = df["low"].values
        c  = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr
