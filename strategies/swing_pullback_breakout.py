"""
strategies/swing_pullback_breakout.py
4H Swing Trend Pullback Breakout

Logic
-----
1. Trend filter: EMA50 slope (up/down) + ADX > 20
2. Pullback memory: price retraces to EMA20 zone without breaking structure
3. Breakout: price closes back above EMA20 (long) or below EMA20 (short)
   after the pullback + RSI > 50 (long) or < 50 (short)
4. SL below pullback low (long) or above pullback high (short)
5. TP = 2.5× risk
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class SwingPullbackBreakout(BaseStrategy):

    name        = "SwingPullbackBreakout"
    description = "4H EMA50 trend + EMA20 pullback + RSI + ADX breakout"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    EMA_FAST    = 20
    EMA_SLOW    = 50
    RSI_PERIOD  = 14
    ADX_PERIOD  = 14
    ADX_MIN     = 20
    PULL_BARS   = 10         # memory: how many bars to look for the pullback
    ATR_MULT_SL = 0.3        # extra buffer below pullback low
    TP_RATIO    = 2.5

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.EMA_SLOW + self.ADX_PERIOD + self.PULL_BARS + 10:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        atr_s  = df["atr"] if "atr" in df.columns else self._calc_atr(df, 14)

        ema20 = close.ewm(span=self.EMA_FAST, adjust=False).mean()
        ema50 = close.ewm(span=self.EMA_SLOW, adjust=False).mean()
        rsi   = self._rsi(close, self.RSI_PERIOD)
        adx   = self._adx(df, self.ADX_PERIOD)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        rsi_v = float(rsi.iloc[i])
        adx_v = float(adx.iloc[i]) if not np.isnan(adx.iloc[i]) else 0.0
        e20   = float(ema20.iloc[i])
        e50   = float(ema50.iloc[i])

        # Trend direction
        trend_up   = e50 > float(ema50.iloc[i - 3]) and entry > e50
        trend_down = e50 < float(ema50.iloc[i - 3]) and entry < e50

        strong_trend = adx_v > self.ADX_MIN

        # Pullback memory: did price touch EMA20 zone in last PULL_BARS bars?
        pb = self.PULL_BARS
        recent_lows  = low.iloc[i - pb: i]
        recent_highs = high.iloc[i - pb: i]
        recent_ema20 = ema20.iloc[i - pb: i]

        pull_touched_long  = (recent_lows  <= recent_ema20 * 1.002).any()
        pull_touched_short = (recent_highs >= recent_ema20 * 0.998).any()

        # Breakout: current bar closed back on the right side of EMA20
        breakout_long  = entry > e20 and float(close.iloc[i - 1]) <= float(ema20.iloc[i - 1])
        breakout_short = entry < e20 and float(close.iloc[i - 1]) >= float(ema20.iloc[i - 1])

        # Pullback low/high for SL
        pb_low  = float(recent_lows.min())
        pb_high = float(recent_highs.max())

        signals = []

        if trend_up and strong_trend and pull_touched_long and breakout_long and rsi_v > 50:
            sl = pb_low - atr_v * self.ATR_MULT_SL
            tp = entry + (entry - sl) * self.TP_RATIO
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.5,
                "zone":        {"high": entry + atr_v * 0.5, "low": e20},
                "pattern_key": "swing_pull_break_bull",
                "strategy":    self.name,
                "notes": (f"Swing pullback breakout long: "
                          f"EMA50 up, ADX={adx_v:.1f}, RSI={rsi_v:.1f}"),
            })

        if trend_down and strong_trend and pull_touched_short and breakout_short and rsi_v < 50:
            sl = pb_high + atr_v * self.ATR_MULT_SL
            tp = entry - (sl - entry) * self.TP_RATIO
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.5,
                "zone":        {"high": e20, "low": entry - atr_v * 0.5},
                "pattern_key": "swing_pull_break_bear",
                "strategy":    self.name,
                "notes": (f"Swing pullback breakout short: "
                          f"EMA50 down, ADX={adx_v:.1f}, RSI={rsi_v:.1f}"),
            })

        return signals

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        up    = high.diff()
        down  = -low.diff()
        plus_dm  = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        hl  = high - low
        hc  = (high - close.shift()).abs()
        lc  = (low  - close.shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        pdi = 100 * plus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        mdi = 100 * minus_dm.rolling(period).mean() / atr.replace(0, np.nan)
        dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        return dx.rolling(period).mean()

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
