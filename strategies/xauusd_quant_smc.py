"""
strategies/xauusd_quant_smc.py
XAUUSD Quant SMC  (v1.0)

Logic
-----
  Multi-timeframe trend alignment:
    D1  bias: close > EMA50(D1)   → bullish D1
    4H  bias: close > EMA50(4H)   → bullish 4H
    trend_aligned: both same direction

  NOTE: This strategy runs on a 15M chart. The D1/4H EMAs are approximated
  from the same DataFrame by using EMA(50 × periodMultiplier):
    D1  ≈ EMA(50 × 96)  = EMA(4800)  bars of 15M  (96 × 15min = 24h)
    4H  ≈ EMA(50 × 16)  = EMA(800)   bars of 15M  (16 × 15min = 4h)
  (These large EMAs use enough history when available; degrade gracefully.)

  BOS (Break of Structure) on 15M:
    Use a rolling 20-bar high/low.
    Bullish  BOS: close crosses above 20-bar high (previous bar was below)
    Bearish  BOS: close crosses below 20-bar low
    bos_occurred: stored as state; lasts until next entry or 20 bars

  Fibonacci 0.618 retracement entry:
    After BOS, track the impulse high (fib_high) and low (fib_low):
      Bullish BOS: fib_low = bos_bar low, fib_high = close at BOS
      Bearish BOS: fib_high = bos_bar high, fib_low = close at BOS
    fib_618 bull = fib_high − (fib_high − fib_low) × 0.618
    fib_618 bear = fib_low  + (fib_high − fib_low) × 0.618

    Long  entry: low  crosses under fib_618 (touches retrace) AND close > fib_618[prev]
    Short entry: high crosses over  fib_618 AND close < fib_618[prev]

  Session filter: 7–13 UTC (context["hour"] if available, else always allowed)

  Max 1 trade/day: tracked via context["trades_today"] if available.

  SL:
    Long:  fib_low  − ATR(14) × 0.5
    Short: fib_high + ATR(14) × 0.5

  TP:
    Long:  entry + (entry − SL) × 1.5
    Short: entry − (SL − entry) × 1.5
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class XAUUSDQuantSMC(BaseStrategy):

    name        = "XAUUSDQuantSMC"
    description = "D1+4H EMA50 trend + 15M BOS + Fib 0.618 retrace entry"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    EMA_PERIOD      = 50
    D1_MULT         = 96      # 15M bars per D1 day
    H4_MULT         = 16      # 15M bars per 4H
    BOS_WINDOW      = 20      # rolling high/low window for BOS
    BOS_EXPIRY      = 20      # bars after BOS before it expires
    FIB_LEVEL       = 0.618
    ATR_LEN         = 14
    ATR_SL_BUF      = 0.5
    TP_RR           = 1.5
    SESSION_START   = 7       # UTC hour
    SESSION_END     = 13      # UTC hour (exclusive)
    MAX_DAILY_TRADES = 1

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        needed = max(self.EMA_PERIOD * self.H4_MULT // 10 + 5,
                     self.BOS_WINDOW + 5, self.ATR_LEN + 5)
        # Be graceful: just need enough for at least H4 EMA
        if len(df) < self.BOS_WINDOW + self.ATR_LEN + 10:
            return []

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        atr_s = df["atr"].values if "atr" in df.columns else self._calc_atr_arr(df)

        i = len(df) - 1
        if np.isnan(atr_s[i]):
            return []

        atr_val = float(atr_s[i])

        # ── Session filter ────────────────────────────────────────────────────
        hour = context.get("hour")
        if hour is not None:
            if not (self.SESSION_START <= int(hour) < self.SESSION_END):
                return []

        # ── Daily trade limit ─────────────────────────────────────────────────
        if context.get("trades_today", 0) >= self.MAX_DAILY_TRADES:
            return []

        # ── Multi-TF trend alignment ──────────────────────────────────────────
        n_h4 = min(self.EMA_PERIOD * self.H4_MULT, i)
        n_d1 = min(self.EMA_PERIOD * self.D1_MULT, i)

        ema_h4 = self._ema_last(close, n_h4)
        ema_d1 = self._ema_last(close, n_d1)

        c_now = float(close[i])
        bull_h4 = c_now > ema_h4
        bull_d1 = c_now > ema_d1
        bear_h4 = c_now < ema_h4
        bear_d1 = c_now < ema_d1

        trend_bull = bull_h4 and bull_d1
        trend_bear = bear_h4 and bear_d1

        if not trend_bull and not trend_bear:
            return []

        # ── BOS detection ─────────────────────────────────────────────────────
        bos_bar, is_bull_bos = self._detect_bos(close, high, low, i)
        if bos_bar is None:
            return []
        if is_bull_bos and not trend_bull:
            return []
        if not is_bull_bos and not trend_bear:
            return []

        # ── Fib 0.618 retrace ─────────────────────────────────────────────────
        if is_bull_bos:
            fib_low  = float(np.min(low[bos_bar: i + 1]))
            fib_high = float(close[bos_bar])
        else:
            fib_high = float(np.max(high[bos_bar: i + 1]))
            fib_low  = float(close[bos_bar])

        fib_618 = (fib_high - (fib_high - fib_low) * self.FIB_LEVEL if is_bull_bos
                   else fib_low  + (fib_high - fib_low) * self.FIB_LEVEL)

        c_prev = float(close[i - 1])
        l_now  = float(low[i])
        h_now  = float(high[i])

        long_entry  = is_bull_bos  and l_now <= fib_618 and c_now > fib_618
        short_entry = not is_bull_bos and h_now >= fib_618 and c_now < fib_618

        if not long_entry and not short_entry:
            return []

        sig_type = "buy" if long_entry else "sell"
        entry    = c_now

        if sig_type == "buy":
            sl = fib_low  - atr_val * self.ATR_SL_BUF
        else:
            sl = fib_high + atr_val * self.ATR_SL_BUF

        risk = abs(entry - sl)
        if risk < 1e-10:
            return []

        tp = (entry + risk * self.TP_RR if sig_type == "buy"
              else entry - risk * self.TP_RR)

        quality = self._quality(sig_type, context, htf_bias)

        return [{
            "type":        sig_type,
            "entry_price": round(entry, 5),
            "sl_price":    round(sl, 5),
            "tp_price":    round(tp, 5),
            "quality":     quality,
            "zone":        {"high": round(fib_618 + atr_val * 0.5, 5),
                            "low":  round(fib_618 - atr_val * 0.5, 5)},
            "pattern_key": f"xau_smc_{sig_type}",
            "strategy":    self.name,
            "notes":       (f"BOS {'bull' if is_bull_bos else 'bear'} | "
                            f"fib618={fib_618:.3f} | "
                            f"fib=[{fib_low:.3f}-{fib_high:.3f}] | "
                            f"H4/D1_EMA50 aligned | session={self.SESSION_START}-{self.SESSION_END}UTC"),
        }]

    # ── BOS detection ─────────────────────────────────────────────────────────

    def _detect_bos(self, close, high, low, i):
        """Scan back BOS_EXPIRY bars for a BOS event."""
        w = self.BOS_WINDOW
        expiry = self.BOS_EXPIRY
        for lag in range(1, expiry + 1):
            k = i - lag
            if k < w:
                break
            roll_high = float(np.max(high[k - w: k]))
            roll_low  = float(np.min(low[k  - w: k]))
            bull_bos = float(close[k - 1]) <= roll_high and float(close[k]) > roll_high
            bear_bos = float(close[k - 1]) >= roll_low  and float(close[k]) < roll_low
            if bull_bos:
                return k, True
            if bear_bos:
                return k, False
        return None, None

    # ── EMA (last value only) ─────────────────────────────────────────────────

    @staticmethod
    def _ema_last(arr: np.ndarray, n: int) -> float:
        if n < 2 or len(arr) < n:
            return float(arr[-1]) if len(arr) > 0 else 0.0
        mult = 2.0 / (n + 1)
        val  = float(arr[:n].mean())
        for v in arr[n:]:
            val = float(v) * mult + val * (1 - mult)
        return val

    # ── ATR ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr_arr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
        h = df["high"].values; l = df["low"].values; c = df["close"].values
        c1 = np.roll(c, 1); c1[0] = c[0]
        tr  = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
        atr = np.full(len(tr), np.nan)
        atr[n - 1] = tr[:n].mean()
        for k in range(n, len(tr)):
            atr[k] = (atr[k - 1] * (n - 1) + tr[k]) / n
        return atr

    # ── Quality ───────────────────────────────────────────────────────────────

    @staticmethod
    def _quality(sig_type: str, context: dict, htf_bias: str) -> float:
        score = 6.0    # multi-TF confirmation base
        if context.get("adx", 0) > 25:
            score += 0.5
        if sig_type == "buy"  and htf_bias == "bullish":
            score += 1.0
        elif sig_type == "sell" and htf_bias == "bearish":
            score += 1.0
        return round(min(max(score, 1.0), 10.0), 1)
