"""
strategies/liberty_elite.py
Bmm Liberty Strategy Elite

Logic
-----
  6-EMA trend ribbon: EMA(8), EMA(13), EMA(21), EMA(34), EMA(55), EMA(89).
  Bull stack : 8 > 13 > 21 > 34 > 55 > 89  (fully ordered, bullish).
  Bear stack : 8 < 13 < 21 < 34 < 55 < 89  (fully ordered, bearish).

  Additional filters (all required):
    RSI 40–70 for buy  |  RSI 30–60 for sell
    MACD histogram positive for buy, negative for sell
    VWAP: close above VWAP for buy, below for sell (if 'vwap' in df.columns)

  Session filter: London and New_York only.

  Entry: current candle close when all conditions met.

  SL : below EMA89 − 0.5×ATR (buy) | above EMA89 + 0.5×ATR (sell)
  TP : entry ± 2×risk

  Quality (base 5):
    +1  RSI ideal range  (50–65 buy  /  35–50 sell)
    +1  MACD strengthening  (|hist[i]| > |hist[i-1]|)
    +1  VWAP aligned
    +1  ADX > 25
    +1  htf_bias matches
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategies.base_strategy import BaseStrategy


class LibertyElite(BaseStrategy):

    name        = "BmmLibertyElite"
    description = "6-EMA ribbon + RSI + MACD + VWAP confluence system"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    EMA_PERIODS  = (8, 13, 21, 34, 55, 89)
    RSI_BUY_LO   = 40;  RSI_BUY_HI  = 70
    RSI_SELL_LO  = 30;  RSI_SELL_HI = 60
    ATR_SL_BUFF  = 0.5
    RR_TARGET    = 2.0
    MIN_BARS     = 100    # need at least EMA89 + buffer

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

        if session not in ("london", "new_york"):
            return []

        close = df["close"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)
        i     = len(df) - 1

        atr_val = float(atr_s[i]) if not np.isnan(atr_s[i]) else float(close[i]) * 0.001
        if atr_val <= 0:
            return []

        # ── 6-EMA ribbon ──────────────────────────────────────────────────────
        emas = {}
        for p in self.EMA_PERIODS:
            e = self._ema(close, p)
            if np.isnan(e[i]):
                return []
            emas[p] = float(e[i])

        bull_stack = all(
            emas[self.EMA_PERIODS[k]] > emas[self.EMA_PERIODS[k + 1]]
            for k in range(len(self.EMA_PERIODS) - 1)
        )
        bear_stack = all(
            emas[self.EMA_PERIODS[k]] < emas[self.EMA_PERIODS[k + 1]]
            for k in range(len(self.EMA_PERIODS) - 1)
        )

        if not bull_stack and not bear_stack:
            return []

        entry = float(close[i])

        # ── RSI filter ────────────────────────────────────────────────────────
        rsi_val = None
        if "rsi" in df.columns:
            rv = df["rsi"].values[i]
            if not np.isnan(rv):
                rsi_val = float(rv)

        if rsi_val is None:
            # no RSI → skip
            return []

        if bull_stack and not (self.RSI_BUY_LO <= rsi_val <= self.RSI_BUY_HI):
            return []
        if bear_stack and not (self.RSI_SELL_LO <= rsi_val <= self.RSI_SELL_HI):
            return []

        # ── MACD filter ───────────────────────────────────────────────────────
        macd_hist    = None
        macd_hist_p  = None
        if "macd_hist" in df.columns:
            mh = df["macd_hist"].values
            if not np.isnan(mh[i]):
                macd_hist = float(mh[i])
            if i > 0 and not np.isnan(mh[i - 1]):
                macd_hist_p = float(mh[i - 1])

        if macd_hist is None:
            return []

        if bull_stack and macd_hist <= 0:
            return []
        if bear_stack and macd_hist >= 0:
            return []

        # ── VWAP filter (optional) ────────────────────────────────────────────
        vwap_ok    = True
        vwap_val   = None
        vwap_aligned = False
        if "vwap" in df.columns:
            vv = df["vwap"].values[i]
            if not np.isnan(vv):
                vwap_val = float(vv)
                if bull_stack and entry <= vwap_val:
                    vwap_ok = False
                if bear_stack and entry >= vwap_val:
                    vwap_ok = False
                if vwap_ok:
                    vwap_aligned = True

        if not vwap_ok:
            return []

        sig_type = "buy" if bull_stack else "sell"
        ema89    = emas[89]

        # ── SL / TP ───────────────────────────────────────────────────────────
        if sig_type == "buy":
            sl = ema89 - self.ATR_SL_BUFF * atr_val
            if sl >= entry:
                sl = entry - 1.5 * atr_val
        else:
            sl = ema89 + self.ATR_SL_BUFF * atr_val
            if sl <= entry:
                sl = entry + 1.5 * atr_val

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = entry + self.RR_TARGET * risk if sig_type == "buy" \
             else entry - self.RR_TARGET * risk

        # ── Quality ───────────────────────────────────────────────────────────
        quality = self._score_quality(
            sig_type, rsi_val, macd_hist, macd_hist_p,
            vwap_aligned, htf_bias, context
        )

        notes = (
            f"EMA ribbon {'bull' if bull_stack else 'bear'} | "
            f"RSI={round(rsi_val, 1)} | "
            f"MACDhist={round(macd_hist, 5)} | "
            f"VWAP={'above' if (vwap_val and entry > vwap_val) else 'below' if vwap_val else 'n/a'} | "
            f"EMA89={round(ema89, 5)}"
        )

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {
                "high": round(float(df["high"].values[i]), 5),
                "low":  round(float(df["low"].values[i]),  5),
            },
            "pattern_key": f"liberty_ema_ribbon_{sig_type}",
            "strategy":    self.name,
            "notes":       notes,
        }]

    # ── quality scoring ───────────────────────────────────────────────────────

    @staticmethod
    def _score_quality(
        sig_type: str,
        rsi_val: float,
        macd_hist: float,
        macd_hist_p,
        vwap_aligned: bool,
        htf_bias: str,
        context: Dict[str, Any],
    ) -> float:
        score = 5.0

        # RSI ideal sub-range
        if sig_type == "buy"  and (50.0 <= rsi_val <= 65.0):
            score += 1.0
        if sig_type == "sell" and (35.0 <= rsi_val <= 50.0):
            score += 1.0

        # MACD strengthening
        if macd_hist_p is not None:
            if abs(macd_hist) > abs(macd_hist_p):
                score += 1.0

        # VWAP aligned
        if vwap_aligned:
            score += 1.0

        # ADX
        if context.get("adx", 0) > 25:
            score += 1.0

        # HTF bias
        bias = context.get("htf_bias", htf_bias)
        if sig_type == "buy"  and bias == "bullish":
            score += 1.0
        if sig_type == "sell" and bias == "bearish":
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
