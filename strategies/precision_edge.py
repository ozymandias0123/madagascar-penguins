"""
strategies/precision_edge.py
Precision Edge System

Logic
-----
  Opening Range Breakout (ORB)
    The "opening range" is the high and low of the first min(30, len-1) bars
    of the current session (bars 0 … 29).

  Fair Value Gap (FVG) — 3-bar pattern
    Bullish FVG : bar[-1].low > bar[-3].high  (gap between them, no overlap)
    Bearish FVG : bar[-1].high < bar[-3].low

  Kill zones : session = 'london' or 'new_york'

  Liquidity sweep (for buy):
    bar[-2].low < bar[-3].low  (wick below prior swing low)
    AND bar[-2].close > bar[-3].low  (reclaimed above, wick rejection)
  Liquidity sweep (for sell): mirror logic with highs.

  RSI-2 layer (mean reversion confirmation):
    buy  confirmed if RSI(2) < 10   (extreme oversold)
    sell confirmed if RSI(2) > 90   (extreme overbought)

  Entry conditions (all required):
    - Session is london or new_york
    - FVG present in the trade direction
    - ORB breakout: close > opening_range_high (buy) | close < opening_range_low (sell)

  SL : min(opening range midpoint distance, 1.5×ATR) beyond entry
  TP : 2×risk  (R:R = 2)

  Quality (base 5):
    +1  FVG confirmed
    +1  Liquidity sweep present
    +1  Kill zone (london / new_york)
    +1  htf_bias aligned
    +1  RSI-2 extreme
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy


class PrecisionEdge(BaseStrategy):

    name        = "PrecisionEdgeSystem"
    description = "ORB breakout + FVG + liquidity sweep + RSI-2 mean reversion"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    ORB_BARS     = 30      # bars that form the opening range
    ATR_SL_MULT  = 1.5
    RR_TARGET    = 2.0
    RSI2_OB      = 90.0   # RSI-2 overbought threshold (sell)
    RSI2_OS      = 10.0   # RSI-2 oversold  threshold (buy)
    MIN_BARS     = 35

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

        # Kill-zone gate
        in_kill_zone = session in ("london", "new_york")
        if not in_kill_zone:
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1

        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else float(close[i]) * 0.001
        if atr_val <= 0:
            return []

        # ── Opening Range ────────────────────────────────────────────────────
        orb_end      = min(self.ORB_BARS, i)
        orb_high     = float(np.max(high[:orb_end]))
        orb_low      = float(np.min(low[:orb_end]))
        orb_mid      = (orb_high + orb_low) / 2.0
        if orb_high <= orb_low:
            return []

        entry = float(close[i])

        orb_bull = entry > orb_high   # bullish breakout
        orb_bear = entry < orb_low    # bearish breakout

        if not orb_bull and not orb_bear:
            return []

        # ── FVG detection ─────────────────────────────────────────────────────
        fvg_bull = self._fvg_bull(high, low, i)
        fvg_bear = self._fvg_bear(high, low, i)

        if orb_bull and not fvg_bull:
            return []
        if orb_bear and not fvg_bear:
            return []

        # ── Liquidity sweep ───────────────────────────────────────────────────
        liq_sweep_bull = self._liq_sweep_bull(low, close, i)
        liq_sweep_bear = self._liq_sweep_bear(high, close, i)

        # ── RSI-2 ─────────────────────────────────────────────────────────────
        rsi2_arr  = self._rsi(close, 2)
        rsi2_val  = float(rsi2_arr[i]) if not np.isnan(rsi2_arr[i]) else 50.0
        rsi2_buy  = rsi2_val < self.RSI2_OS
        rsi2_sell = rsi2_val > self.RSI2_OB

        sig_type = "buy" if orb_bull else "sell"

        # ── SL: tighter of ORB-midpoint distance or 1.5×ATR ──────────────────
        if sig_type == "buy":
            sl_atr  = entry - self.ATR_SL_MULT * atr_val
            sl_orb  = orb_mid
            sl      = max(sl_atr, sl_orb)    # closest SL (less risk)
        else:
            sl_atr  = entry + self.ATR_SL_MULT * atr_val
            sl_orb  = orb_mid
            sl      = min(sl_atr, sl_orb)

        risk = abs(entry - sl)
        if risk < 1e-10:
            sl   = entry - 1.5 * atr_val if sig_type == "buy" else entry + 1.5 * atr_val
            risk = abs(entry - sl)

        tp = entry + self.RR_TARGET * risk if sig_type == "buy" \
             else entry - self.RR_TARGET * risk

        # ── Quality ───────────────────────────────────────────────────────────
        liq_ok   = liq_sweep_bull if sig_type == "buy" else liq_sweep_bear
        rsi2_ok  = rsi2_buy       if sig_type == "buy" else rsi2_sell
        quality  = self._score_quality(sig_type, fvg_bull or fvg_bear,
                                       liq_ok, in_kill_zone,
                                       rsi2_ok, htf_bias, context)

        notes = (
            f"ORB break {'bull' if orb_bull else 'bear'} | "
            f"ORB range [{round(orb_low,5)}–{round(orb_high,5)}] | "
            f"FVG={fvg_bull if sig_type=='buy' else fvg_bear} | "
            f"LiqSweep={liq_ok} | RSI2={round(rsi2_val,1)} | "
            f"session={session}"
        )

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {
                "high": round(orb_high, 5),
                "low":  round(orb_low,  5),
            },
            "pattern_key": f"precision_orb_{sig_type}",
            "strategy":    self.name,
            "notes":       notes,
        }]

    # ── FVG / liquidity helpers ───────────────────────────────────────────────

    @staticmethod
    def _fvg_bull(high: np.ndarray, low: np.ndarray, i: int) -> bool:
        """Bullish FVG: gap between bar[-3].high and bar[-1].low."""
        if i < 2:
            return False
        return float(low[i]) > float(high[i - 2])

    @staticmethod
    def _fvg_bear(high: np.ndarray, low: np.ndarray, i: int) -> bool:
        """Bearish FVG: gap between bar[-3].low and bar[-1].high."""
        if i < 2:
            return False
        return float(high[i]) < float(low[i - 2])

    @staticmethod
    def _liq_sweep_bull(low: np.ndarray, close: np.ndarray, i: int) -> bool:
        """Wick below prior swing low then reclaimed above it."""
        if i < 2:
            return False
        prior_low = float(low[i - 2])
        return (float(low[i - 1]) < prior_low) and (float(close[i - 1]) > prior_low)

    @staticmethod
    def _liq_sweep_bear(high: np.ndarray, close: np.ndarray, i: int) -> bool:
        """Wick above prior swing high then reclaimed below it."""
        if i < 2:
            return False
        prior_high = float(high[i - 2])
        return (float(high[i - 1]) > prior_high) and (float(close[i - 1]) < prior_high)

    @staticmethod
    def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
        out = np.full(len(close), np.nan)
        if len(close) < n + 1:
            return out
        delta = np.diff(close, prepend=close[0])
        gain  = np.where(delta > 0, delta, 0.0)
        loss  = np.where(delta < 0, -delta, 0.0)
        avg_g = np.full(len(close), np.nan)
        avg_l = np.full(len(close), np.nan)
        avg_g[n] = gain[1: n + 1].mean()
        avg_l[n] = loss[1: n + 1].mean()
        for k in range(n + 1, len(close)):
            avg_g[k] = (avg_g[k - 1] * (n - 1) + gain[k]) / n
            avg_l[k] = (avg_l[k - 1] * (n - 1) + loss[k]) / n
        rs       = np.where(avg_l > 0, avg_g / avg_l, 100.0)
        out[n:]  = 100.0 - 100.0 / (1.0 + rs[n:])
        return out

    # ── quality scoring ───────────────────────────────────────────────────────

    @staticmethod
    def _score_quality(
        sig_type: str,
        fvg: bool,
        liq_sweep: bool,
        kill_zone: bool,
        rsi2_extreme: bool,
        htf_bias: str,
        context: Dict[str, Any],
    ) -> float:
        score = 5.0
        if fvg:
            score += 1.0
        if liq_sweep:
            score += 1.0
        if kill_zone:
            score += 1.0
        bias = context.get("htf_bias", htf_bias)
        if sig_type == "buy"  and bias == "bullish":
            score += 1.0
        if sig_type == "sell" and bias == "bearish":
            score += 1.0
        if rsi2_extreme:
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
