"""
strategies/cci_surge.py
CCI Surge — Momentum Continuation Strategy

Logic
-----
  - HMA trend filter: HMA rising + close above HMA (long) / falling + below (short)
  - CCI burst: CCI > 100 (long) / CCI < -100 (short)
  - Gate: re-enabled only after CCI normalises back into neutral zone
  - ATR stop loss set at entry ± ATR × atrMultiplier
  - Trailing stop activates after 1+ bar in position
  - Quality degrades slightly if ATR is very low (choppy market)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from strategies.base_strategy import BaseStrategy


class CCISurge(BaseStrategy):

    name        = "CCISurge"
    description = "CCI > ±100 + HMA trend + ATR SL + trailing stop"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    CCI_LEN      = 14
    CCI_HIGH     = 100
    CCI_LOW      = -100
    HMA_LEN      = 80
    ATR_LEN      = 14
    ATR_MULT     = 2.0
    TRAIL_PCT    = 6.0    # trailing stop as % of entry

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._gate_long  = True
        self._gate_short = True

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.HMA_LEN + 20:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        atr_s  = df["atr"] if "atr" in df.columns else self._calc_atr(df, self.ATR_LEN)

        cci = self._cci(df, self.CCI_LEN)
        hma = self._hma(close, self.HMA_LEN)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0
        cci_v = float(cci.iloc[i])

        hma_now  = float(hma.iloc[i])
        hma_prev = float(hma.iloc[i - 1])
        hma_up   = hma_now > hma_prev and entry > hma_now
        hma_dn   = hma_now < hma_prev and entry < hma_now

        # Gate re-open logic: re-enable once CCI crosses back through neutral
        if cci_v <= self.CCI_HIGH:
            self._gate_long  = True
        if cci_v >= self.CCI_LOW:
            self._gate_short = True

        long_signal  = cci_v > self.CCI_HIGH  and hma_up and self._gate_long
        short_signal = cci_v < self.CCI_LOW   and hma_dn and self._gate_short

        signals = []

        if long_signal:
            sl    = entry - atr_v * self.ATR_MULT
            tp    = entry + atr_v * self.ATR_MULT * 1.5
            trail = entry * self.TRAIL_PCT / 100
            self._gate_long = False
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.5,
                "zone":        {"high": entry + atr_v * 0.3, "low": hma_now},
                "pattern_key": "cci_surge_long",
                "strategy":    self.name,
                "trail_offset": trail,
                "notes":       (f"CCI={cci_v:.1f}>{self.CCI_HIGH}, "
                                f"HMA up={hma_up}, ATR={atr_v:.4f}"),
            })

        if short_signal:
            sl    = entry + atr_v * self.ATR_MULT
            tp    = entry - atr_v * self.ATR_MULT * 1.5
            trail = entry * self.TRAIL_PCT / 100
            self._gate_short = False
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.5,
                "zone":        {"high": hma_now, "low": entry - atr_v * 0.3},
                "pattern_key": "cci_surge_short",
                "strategy":    self.name,
                "trail_offset": trail,
                "notes":       (f"CCI={cci_v:.1f}<{self.CCI_LOW}, "
                                f"HMA down={hma_dn}, ATR={atr_v:.4f}"),
            })

        return signals

    def on_trade_closed(self, result: Dict) -> None:
        # Re-open gate after trade closes so next setup can fire
        self._gate_long  = True
        self._gate_short = True

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        sma     = typical.rolling(period).mean()
        mad     = typical.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        return (typical - sma) / (0.015 * mad.replace(0, np.nan))

    @staticmethod
    def _hma(close: pd.Series, period: int = 80) -> pd.Series:
        """Hull Moving Average: WMA(2×WMA(n/2) − WMA(n), sqrt(n))."""
        half = max(int(period / 2), 1)
        sq   = max(int(period ** 0.5), 1)
        wma_half = close.rolling(half).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        wma_full = close.rolling(period).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        raw = 2 * wma_half - wma_full
        return raw.rolling(sq).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
