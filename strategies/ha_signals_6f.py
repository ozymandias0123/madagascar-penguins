"""
strategies/ha_signals_6f.py
6F Signals + EMA / MA Filter

Logic
-----
  - Heikin Ashi candles computed from raw OHLC
  - Color-change detection: HA flips bullish → record c_high/c_low
  - Signal: HA close breaks above c_high (buy) or below c_low (sell)
    after a color flip — one signal per flip
  - MA filter: MA6 > MA9 required for buy / MA6 < MA9 for sell
  - SL: HA low of previous bar (buy) / HA high of previous bar (sell)
         buffered by sl_buffer_pct
  - TP: entry × (1 ± tp_pct)
  - Leverage: implied from SL distance (encoded in quality)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional
from strategies.base_strategy import BaseStrategy


class HASignals6F(BaseStrategy):

    name        = "HASignals6F"
    description = "Heikin Ashi color flip + MA6/MA9 filter + TP/SL"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    MA_FAST      = 6
    MA_SLOW      = 9
    TP_PCT       = 0.03      # 3%
    SL_BUFFER    = 0.01      # 1%
    MIN_LEV      = 1
    MAX_LEV      = 20

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        if len(df) < self.MA_SLOW + 10:
            return []

        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        open_ = df["open"]
        atr_s = df["atr"] if "atr" in df.columns else self._calc_atr(df, 14)

        # ── Heikin Ashi ───────────────────────────────────────────────────────
        ha_close = (open_ + high + low + close) / 4
        ha_open  = pd.Series(np.nan, index=df.index)
        ha_open.iloc[0] = float(open_.iloc[0])
        for k in range(1, len(df)):
            ha_open.iloc[k] = (float(ha_open.iloc[k - 1]) + float(ha_close.iloc[k - 1])) / 2
        ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
        ha_low  = pd.concat([low,  ha_open, ha_close], axis=1).min(axis=1)
        ha_bull = ha_close > ha_open   # True = bullish HA candle

        # ── MA filter ─────────────────────────────────────────────────────────
        ma6 = close.rolling(self.MA_FAST).mean()
        ma9 = close.rolling(self.MA_SLOW).mean()
        bull_trend = ma6 > ma9
        bear_trend = ma6 < ma9

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 10.0

        # ── Find last HA color flip ───────────────────────────────────────────
        # Walk backwards from i-1 to find the most recent flip
        c_high: Optional[float] = None
        c_low:  Optional[float] = None
        flip_bar: Optional[int] = None
        signal_used = False

        for k in range(i - 1, max(i - 30, -len(df)) - 1, -1):
            if bool(ha_bull.iloc[k]) != bool(ha_bull.iloc[k - 1]):
                c_high    = float(ha_high.iloc[k])
                c_low     = float(ha_low.iloc[k])
                flip_bar  = k
                break

        if c_high is None or c_low is None:
            return []

        # ── Signal conditions ─────────────────────────────────────────────────
        # Buy:  current HA close > c_high AND MA filter bullish
        # Sell: current HA close < c_low  AND MA filter bearish
        buy_raw  = float(ha_close.iloc[i]) > c_high
        sell_raw = float(ha_close.iloc[i]) < c_low

        buy_signal  = buy_raw  and bool(bull_trend.iloc[i])
        sell_signal = sell_raw and bool(bear_trend.iloc[i])

        signals = []

        if buy_signal:
            sl     = float(ha_low.iloc[i - 1]) * (1 - self.SL_BUFFER)
            tp     = entry * (1 + self.TP_PCT)
            lev    = self._implied_leverage(entry, sl)
            q      = min(5.0 + lev * 0.2, 9.0)
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     q,
                "zone":        {"high": c_high, "low": c_low},
                "pattern_key": "ha_6f_buy",
                "strategy":    self.name,
                "notes":       (f"HA flip buy: close>{c_high:.4f}, "
                                f"MA6>MA9={bool(bull_trend.iloc[i])}, "
                                f"lev~{lev}x"),
            })

        if sell_signal:
            sl  = float(ha_high.iloc[i - 1]) * (1 + self.SL_BUFFER)
            tp  = entry * (1 - self.TP_PCT)
            lev = self._implied_leverage(entry, sl)
            q   = min(5.0 + lev * 0.2, 9.0)
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     q,
                "zone":        {"high": c_high, "low": c_low},
                "pattern_key": "ha_6f_sell",
                "strategy":    self.name,
                "notes":       (f"HA flip sell: close<{c_low:.4f}, "
                                f"MA6<MA9={bool(bear_trend.iloc[i])}, "
                                f"lev~{lev}x"),
            })

        return signals

    # ── helpers ───────────────────────────────────────────────────────────────

    def _implied_leverage(self, entry: float, sl: float) -> int:
        if entry == sl:
            return self.MIN_LEV
        dist = abs(entry - sl)
        lev  = round(entry / dist)
        return max(self.MIN_LEV, min(lev, self.MAX_LEV))

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["high"] - df["low"]
        hc  = (df["high"] - df["close"].shift()).abs()
        lc  = (df["low"]  - df["close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()
