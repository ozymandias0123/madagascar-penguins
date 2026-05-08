"""
strategies/pa_engulfing_ob.py
Price Action — Engulfing + Order Block

Logic
-----
  Classic price-action entry using:
    1. Engulfing candle pattern (strong directional momentum)
    2. Order Block (OB) as the entry zone — the last opposing candle
       before the impulse that created the engulf
    3. EMA trend filter (EMA50 direction)
    4. Volume confirmation: engulfing candle volume > 1.5× average (if available)

  Order Block definition:
    Bullish OB: the last bearish candle BEFORE a bullish engulf
      OB zone = [OB candle low, OB candle high]
    Bearish OB: the last bullish candle BEFORE a bearish engulf
      OB zone = [OB candle low, OB candle high]

  Price must retrace into the OB on the current bar (or have just swept it).

  SL : OB low - 0.3×ATR (buy) / OB high + 0.3×ATR (sell)
  TP : 2.5×risk
  Quality: 6 base, +1 volume spike, +1 EMA50 aligned, +1 htf_bias, +1 ADX>22
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class PAEngulfingOB(BaseStrategy):

    name        = "PAEngulfingOB"
    description = "Engulfing candle + Order Block retest entry"
    version     = "1.0"

    EMA_LEN    = 50
    VOL_MULT   = 1.5
    SL_BUF     = 0.3
    RR         = 2.5
    LOOK_BACK  = 5   # bars to search for OB before the engulf

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
        vol_s  = df["volume"].values if "volume" in df.columns else None

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        ema50 = self._ema(close, self.EMA_LEN)
        if np.isnan(ema50[i]):
            return []

        entry   = float(close[i])
        atr_val = float(atr_s[i])
        signals = []

        # ── Bullish engulfing at OB ───────────────────────────────────────────
        if i >= 2:
            # Current bar: bullish and body > prev bar body
            curr_bull = float(close[i]) > float(open_[i])
            curr_body = abs(float(close[i]) - float(open_[i]))
            prev_body = abs(float(close[i-1]) - float(open_[i-1]))
            prev_bear = float(close[i-1]) < float(open_[i-1])
            engulf_bull = curr_bull and prev_bear and curr_body > prev_body

            if engulf_bull and entry > ema50[i]:
                # Find bullish OB: last bearish candle before the engulf
                ob_high, ob_low = None, None
                for j in range(i - 2, max(i - self.LOOK_BACK - 2, 0), -1):
                    if float(close[j]) < float(open_[j]):   # bearish candle
                        ob_high = float(high[j])
                        ob_low  = float(low[j])
                        break
                if ob_high is not None:
                    # Price must be at or have just swept the OB
                    if float(low[i]) <= ob_high:
                        vol_ok  = self._vol_spike(vol_s, i, self.VOL_MULT)
                        sl      = ob_low - self.SL_BUF * atr_val
                        risk    = abs(entry - sl)
                        if risk > 1e-10:
                            tp   = entry + self.RR * risk
                            qual = self._quality(vol_ok, ema50[i], entry, htf_bias,
                                                 "buy", context)
                            signals.append({
                                "type":        "buy",
                                "entry_price": round(entry, 5),
                                "sl_price":    round(sl, 5),
                                "tp_price":    round(tp, 5),
                                "quality":     qual,
                                "zone":        {"high": round(ob_high, 5),
                                                "low":  round(ob_low, 5)},
                                "pattern_key": "pa_engulf_ob_buy",
                                "strategy":    self.name,
                                "notes":       (f"Bull engulf | OB [{round(ob_low,5)}-{round(ob_high,5)}] | "
                                                f"EMA50={'above' if entry>ema50[i] else 'below'}"),
                            })

        # ── Bearish engulfing at OB ───────────────────────────────────────────
        if i >= 2:
            curr_bear = float(close[i]) < float(open_[i])
            curr_body = abs(float(close[i]) - float(open_[i]))
            prev_body = abs(float(close[i-1]) - float(open_[i-1]))
            prev_bull = float(close[i-1]) > float(open_[i-1])
            engulf_bear = curr_bear and prev_bull and curr_body > prev_body

            if engulf_bear and entry < ema50[i]:
                ob_high2, ob_low2 = None, None
                for j in range(i - 2, max(i - self.LOOK_BACK - 2, 0), -1):
                    if float(close[j]) > float(open_[j]):   # bullish candle
                        ob_high2 = float(high[j])
                        ob_low2  = float(low[j])
                        break
                if ob_high2 is not None:
                    if float(high[i]) >= ob_low2:
                        vol_ok  = self._vol_spike(vol_s, i, self.VOL_MULT)
                        sl      = ob_high2 + self.SL_BUF * atr_val
                        risk    = abs(sl - entry)
                        if risk > 1e-10:
                            tp   = entry - self.RR * risk
                            qual = self._quality(vol_ok, ema50[i], entry, htf_bias,
                                                 "sell", context)
                            signals.append({
                                "type":        "sell",
                                "entry_price": round(entry, 5),
                                "sl_price":    round(sl, 5),
                                "tp_price":    round(tp, 5),
                                "quality":     qual,
                                "zone":        {"high": round(ob_high2, 5),
                                                "low":  round(ob_low2, 5)},
                                "pattern_key": "pa_engulf_ob_sell",
                                "strategy":    self.name,
                                "notes":       (f"Bear engulf | OB [{round(ob_low2,5)}-{round(ob_high2,5)}] | "
                                                f"EMA50={'below' if entry<ema50[i] else 'above'}"),
                            })

        return signals

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _vol_spike(vol, i, mult) -> bool:
        if vol is None or i < 20:
            return False
        avg = float(np.nanmean(vol[i - 20: i]))
        return avg > 0 and float(vol[i]) > avg * mult

    @staticmethod
    def _quality(vol_ok, ema, entry, htf_bias, sig_type, context) -> float:
        score = 6.0
        if vol_ok: score += 1.0
        ema_aligned = (sig_type == "buy"  and entry > ema) or \
                      (sig_type == "sell" and entry < ema)
        if ema_aligned: score += 1.0
        bias_ok = (sig_type == "buy"  and htf_bias == "bullish") or \
                  (sig_type == "sell" and htf_bias == "bearish")
        if bias_ok: score += 1.0
        if context.get("adx", 0) > 22: score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)

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
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h  = df["high"].values; l = df["low"].values; c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr
