"""
strategies/breakout_follow_trend.py
Breakout Follow Trend

Logic
-----
1. Bollinger Band breakout: close > upper band (long) / close < lower band (short)
2. EMA trend filter: price above EMA50 (long) / below EMA50 (short)
3. Volume confirmation: volume > 1.5× 20-bar SMA
4. Daily loss limit: strategy suppresses signals if intraday loss exceeds cap
5. Quality degrades after consecutive losses within the same day
"""

import numpy as np
import pandas as pd
from datetime import date
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class BreakoutFollowTrend(BaseStrategy):

    name        = "BreakoutFollowTrend"
    description = "BB breakout + EMA50 + volume filter + daily loss cap"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    BB_PERIOD    = 20
    BB_STD       = 2.0
    EMA_PERIOD   = 50
    VOL_MULT     = 1.5
    ATR_MULT_SL  = 1.2
    TP_RATIO     = 2.0
    DAILY_LOSS_CAP = 3       # max losses per day before halting
    BASE_QUALITY = 7.0

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._daily_losses:  int  = 0
        self._last_trade_day: Any = None

    # ── main ─────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.BB_PERIOD + self.EMA_PERIOD + 5:
            return []

        # Daily loss cap check
        today = date.today()
        if self._last_trade_day != today:
            self._daily_losses   = 0
            self._last_trade_day = today

        if self._daily_losses >= self.DAILY_LOSS_CAP:
            return []   # halted for today

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"]   if "atr"    in df.columns else self._calc_atr(df, 14)

        # Bollinger Bands
        bb_mid  = close.rolling(self.BB_PERIOD).mean()
        bb_std  = close.rolling(self.BB_PERIOD).std()
        bb_up   = bb_mid + self.BB_STD * bb_std
        bb_dn   = bb_mid - self.BB_STD * bb_std

        ema50   = close.ewm(span=self.EMA_PERIOD, adjust=False).mean()
        vol_sma = volume.rolling(20).mean().replace(0, 1)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        bb_u  = float(bb_up.iloc[i])
        bb_d  = float(bb_dn.iloc[i])
        e50   = float(ema50.iloc[i])
        vr    = float(volume.iloc[i]) / float(vol_sma.iloc[i])

        vol_ok     = vr > self.VOL_MULT
        trend_up   = entry > e50
        trend_down = entry < e50
        bo_long    = entry > bb_u
        bo_short   = entry < bb_d

        # Quality degrades with daily loss count
        quality = max(self.BASE_QUALITY - self._daily_losses * 0.5, 4.0)

        signals = []

        if bo_long and trend_up and vol_ok:
            sl = entry - atr_v * self.ATR_MULT_SL
            tp = entry + (entry - sl) * self.TP_RATIO
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     quality,
                "zone":        {"high": bb_u + atr_v * 0.2, "low": bb_mid.iloc[i]},
                "pattern_key": "bb_breakout_long",
                "strategy":    self.name,
                "notes": (f"BB breakout long: price>{bb_u:.2f}, "
                          f"EMA50 up, vol={vr:.1f}x, daily_loss={self._daily_losses}"),
            })

        if bo_short and trend_down and vol_ok:
            sl = entry + atr_v * self.ATR_MULT_SL
            tp = entry - (sl - entry) * self.TP_RATIO
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     quality,
                "zone":        {"high": float(bb_mid.iloc[i]), "low": bb_d - atr_v * 0.2},
                "pattern_key": "bb_breakout_short",
                "strategy":    self.name,
                "notes": (f"BB breakout short: price<{bb_d:.2f}, "
                          f"EMA50 down, vol={vr:.1f}x, daily_loss={self._daily_losses}"),
            })

        return signals

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_trade_closed(self, result: Dict) -> None:
        if result.get("pnl", 0) < 0:
            today = date.today()
            if self._last_trade_day != today:
                self._daily_losses   = 0
                self._last_trade_day = today
            self._daily_losses += 1

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
