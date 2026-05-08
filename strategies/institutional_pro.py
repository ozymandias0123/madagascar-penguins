"""
strategies/institutional_pro.py
Combined Institutional Pro Manager

Fair Value Gaps (FVG) + Break of Structure (BOS) + Volume profile +
Trend filter + State machine (0/1/-1/2/-2) + Break-Even logic.

State machine
-------------
 0  → flat (waiting)
 1  → long setup detected
-1  → short setup detected
 2  → long confirmed + breakeven moved
-2  → short confirmed + breakeven moved
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from strategies.base_strategy import BaseStrategy


class InstitutionalPro(BaseStrategy):

    name        = "InstitutionalPro"
    description = "FVG + BOS + Volume + Trend + state machine + break-even"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    FVG_LOOKBACK   = 3       # bars to look back for FVG
    BOS_LOOKBACK   = 20      # bars for structure pivot
    VOL_MA_LEN     = 20
    VOL_MULT       = 1.5     # volume spike threshold
    TREND_EMA      = 50
    ATR_MULT_SL    = 1.5
    ATR_MULT_TP    = 3.0
    BE_TRIGGER_R   = 1.0     # move BE after 1R

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._state: int = 0

    # ── main ─────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.BOS_LOOKBACK + self.TREND_EMA + 5:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)
        atr_s  = df["atr"]   if "atr"    in df.columns else self._calc_atr(df, 14)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        # ── Trend filter ─────────────────────────────────────────────────────
        ema_trend = close.ewm(span=self.TREND_EMA, adjust=False).mean()
        trend_up   = entry > float(ema_trend.iloc[i])
        trend_down = entry < float(ema_trend.iloc[i])

        # ── Volume spike ─────────────────────────────────────────────────────
        vol_ma    = volume.rolling(self.VOL_MA_LEN).mean()
        vol_spike = float(volume.iloc[i]) > float(vol_ma.iloc[i]) * self.VOL_MULT

        # ── Break of Structure ────────────────────────────────────────────────
        n = self.BOS_LOOKBACK
        recent_high = float(high.iloc[i - n: i].max())
        recent_low  = float(low.iloc[i - n: i].min())
        bos_bull    = float(close.iloc[i]) > recent_high
        bos_bear    = float(close.iloc[i]) < recent_low

        # ── Fair Value Gap ────────────────────────────────────────────────────
        # Bullish FVG: candle[-3] high < candle[-1] low  (gap up)
        # Bearish FVG: candle[-3] low  > candle[-1] high (gap down)
        fvg_bull = float(high.iloc[i - 3]) < float(low.iloc[i - 1])
        fvg_bear = float(low.iloc[i - 3])  > float(high.iloc[i - 1])

        # ── State machine ─────────────────────────────────────────────────────
        # Transition: 0 → 1 or -1 on setup; 1 → 2 or -1 → -2 on confirmation
        if self._state == 0:
            if trend_up and fvg_bull:
                self._state = 1
            elif trend_down and fvg_bear:
                self._state = -1

        if self._state == 1 and bos_bull and vol_spike:
            self._state = 2
        elif self._state == -1 and bos_bear and vol_spike:
            self._state = -2

        signals = []

        if self._state == 2:
            sl = entry - atr_v * self.ATR_MULT_SL
            tp = entry + atr_v * self.ATR_MULT_TP
            be = entry + atr_v * self.BE_TRIGGER_R   # price at which BE triggers
            signals.append({
                "type":           "buy",
                "entry_price":    entry,
                "sl_price":       sl,
                "tp_price":       tp,
                "quality":        8.0,
                "zone":           {"high": recent_high, "low": recent_low},
                "pattern_key":    "inst_pro_bull",
                "strategy":       self.name,
                "breakeven_trigger": be,
                "notes": (f"FVG bull + BOS + vol spike, trend_up, "
                          f"state={self._state}"),
            })
            self._state = 0   # reset after signal

        elif self._state == -2:
            sl = entry + atr_v * self.ATR_MULT_SL
            tp = entry - atr_v * self.ATR_MULT_TP
            be = entry - atr_v * self.BE_TRIGGER_R
            signals.append({
                "type":           "sell",
                "entry_price":    entry,
                "sl_price":       sl,
                "tp_price":       tp,
                "quality":        8.0,
                "zone":           {"high": recent_high, "low": recent_low},
                "pattern_key":    "inst_pro_bear",
                "strategy":       self.name,
                "breakeven_trigger": be,
                "notes": (f"FVG bear + BOS + vol spike, trend_down, "
                          f"state={self._state}"),
            })
            self._state = 0

        return signals

    # ── lifecycle hooks ───────────────────────────────────────────────────────

    def on_trade_closed(self, result: Dict) -> None:
        # Reset state machine if trade closed at a loss
        if result.get("pnl", 0) < 0:
            self._state = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
