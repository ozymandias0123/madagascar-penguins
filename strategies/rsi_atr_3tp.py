"""
strategies/rsi_atr_3tp.py
3-TP RSI-ATR Strategy  [4H]

Logic
-----
  - Trend filter: SMA100 > SMA200 (bull) / SMA100 < SMA200 (bear)
  - Entry:  RSI > 70 + bull trend  → long
            RSI < 30 + bear trend  → short
  - ATR stop loss placed at entry ± ATR × multiplier
  - 3-tier exits encoded as distinct tp_price / tp2_price / tp3_price keys
    so the engine can scale out at each level:
      TP1  10% move from entry  on 25% of position
      TP2  20% move from entry  on 50% of position
      TP3  ATR-based full exit  on remainder
  - Close on opposing RSI signal OR stop hit
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class RSIATRThreeTP(BaseStrategy):

    name        = "RSIATRThreeTP"
    description = "RSI OB/OS + SMA trend + ATR SL + 3-tier TP — 4H"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    RSI_LEN       = 14
    RSI_OB        = 70       # overbought  → long entry
    RSI_OS        = 30       # oversold    → short entry
    SMA_FAST      = 100
    SMA_SLOW      = 200
    ATR_LEN       = 14
    ATR_MULT      = 1.5
    TP1_PCT       = 0.10     # 10%
    TP2_PCT       = 0.20     # 20%
    TP1_QTY_PCT   = 25       # % of position to close at TP1
    TP2_QTY_PCT   = 50       # % of position to close at TP2
    ATR_TP_MULT   = 4.0      # ATR-based TP3 multiplier

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.SMA_SLOW + 20:
            return []

        close = df["close"]
        atr_s = df["atr"] if "atr" in df.columns else self._calc_atr(df, self.ATR_LEN)
        rsi   = self._rsi(close, self.RSI_LEN)
        sma1  = close.rolling(self.SMA_FAST).mean()
        sma2  = close.rolling(self.SMA_SLOW).mean()

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        r     = float(rsi.iloc[i])
        s1    = float(sma1.iloc[i])
        s2    = float(sma2.iloc[i])

        bull_trend  = s1 > s2
        bear_trend  = s1 < s2
        long_cond   = r > self.RSI_OB and bull_trend
        short_cond  = r < self.RSI_OS and bear_trend

        signals = []

        if long_cond:
            sl  = entry - atr_v * self.ATR_MULT
            tp1 = entry * (1 + self.TP1_PCT)
            tp2 = entry * (1 + self.TP2_PCT)
            tp3 = entry + atr_v * self.ATR_TP_MULT
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp3,       # primary TP used by engine
                "tp1_price":   tp1,       # engine can use these for partial exits
                "tp2_price":   tp2,
                "tp1_qty_pct": self.TP1_QTY_PCT,
                "tp2_qty_pct": self.TP2_QTY_PCT,
                "quality":     7.0,
                "zone":        {"high": entry + atr_v * 0.5, "low": sl},
                "pattern_key": "rsi_atr_3tp_long",
                "strategy":    self.name,
                "notes":       (f"RSI={r:.1f}>{self.RSI_OB}, SMA bull, "
                                f"TP1={tp1:.2f} TP2={tp2:.2f} TP3={tp3:.2f}, "
                                f"SL={sl:.2f}"),
            })

        if short_cond:
            sl  = entry + atr_v * self.ATR_MULT
            tp1 = entry * (1 - self.TP1_PCT)
            tp2 = entry * (1 - self.TP2_PCT)
            tp3 = entry - atr_v * self.ATR_TP_MULT
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp3,
                "tp1_price":   tp1,
                "tp2_price":   tp2,
                "tp1_qty_pct": self.TP1_QTY_PCT,
                "tp2_qty_pct": self.TP2_QTY_PCT,
                "quality":     7.0,
                "zone":        {"high": sl, "low": entry - atr_v * 0.5},
                "pattern_key": "rsi_atr_3tp_short",
                "strategy":    self.name,
                "notes":       (f"RSI={r:.1f}<{self.RSI_OS}, SMA bear, "
                                f"TP1={tp1:.2f} TP2={tp2:.2f} TP3={tp3:.2f}, "
                                f"SL={sl:.2f}"),
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
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
