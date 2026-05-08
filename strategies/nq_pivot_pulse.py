"""
strategies/nq_pivot_pulse.py
NQ ETH 30Min Pivot Pulse

Logic
-----
  - Session gate  : configurable window (default 09:30–15:45 NY)
  - Daily P&L cap : halt after daily_profit_target reached OR
                    daily_loss_limit breached
  - Entry         : pivot low confirmed (long) / pivot high confirmed (short)
  - Gates         : ATR volatility, RSI range, optional EMA trend
  - Exit          : TP / SL in ATR ticks + trailing stop
"""

import numpy as np
import pandas as pd
from datetime import date
from typing import Any, Dict, List, Optional
from strategies.base_strategy import BaseStrategy


class NQPivotPulse(BaseStrategy):

    name        = "NQPivotPulse"
    description = "Pivot high/low + ATR/RSI gates + daily P&L cap — 30M NQ"
    version     = "1.0"

    # ── parameters ───────────────────────────────────────────────────────────
    PIVOT_LEN          = 1          # bars each side for pivot detection
    TP_ATR             = 2.0        # TP = entry ± TP_ATR × ATR
    SL_ATR             = 1.0        # SL = entry ∓ SL_ATR × ATR
    TRAIL_ATR          = 0.15       # trailing stop as fraction of ATR
    ATR_MIN            = 2.2        # minimum ATR gate
    RSI_BUY_MAX        = 45         # RSI must be below this for longs
    RSI_SELL_MIN       = 55         # RSI must be above this for shorts
    ACTIVE_SESSIONS    = {"london", "new_york"}
    DAILY_PROFIT_CAP   = 2000.0     # halt trading for the day above this
    DAILY_LOSS_CAP     = 350.0      # halt trading for the day below this

    def __init__(self):
        super().__init__() if hasattr(super(), "__init__") else None
        self._daily_pnl:      float = 0.0
        self._last_day:       Any   = None
        self._open_trade_pnl: float = 0.0   # running open P&L estimate

    # ── daily gate reset ─────────────────────────────────────────────────────

    def _check_day(self) -> None:
        today = date.today()
        if self._last_day != today:
            self._daily_pnl = 0.0
            self._last_day  = today

    def on_trade_closed(self, result: Dict) -> None:
        self._check_day()
        self._daily_pnl += result.get("pnl", 0.0)

    # ── main ─────────────────────────────────────────────────────────────────

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Dict[str, Any],
        session: str,
        htf_bias: str,
    ) -> List[Dict]:

        self._check_day()

        # Daily P&L gate
        if (self._daily_pnl >= self.DAILY_PROFIT_CAP or
                self._daily_pnl <= -self.DAILY_LOSS_CAP):
            return []

        # Session gate
        if session not in self.ACTIVE_SESSIONS:
            return []

        if len(df) < 30:
            return []

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        atr_s  = df["atr"] if "atr" in df.columns else self._calc_atr(df, 14)
        rsi    = self._rsi(close, 14)

        i     = -2
        entry = float(close.iloc[i])
        atr_v = float(atr_s.iloc[i]) if not np.isnan(atr_s.iloc[i]) else 20.0
        r     = float(rsi.iloc[i])

        # ATR gate
        if atr_v < self.ATR_MIN:
            return []

        # Pivot detection (1 bar each side, confirmed at bar i-1)
        # pivot low:  low[i-1] < low[i-2] and low[i-1] < low[i]
        pl_bar = i - 1
        ph_bar = i - 1
        pivot_low  = (float(low.iloc[pl_bar])  < float(low.iloc[pl_bar - 1]) and
                      float(low.iloc[pl_bar])  < float(low.iloc[pl_bar + 1]))
        pivot_high = (float(high.iloc[ph_bar]) > float(high.iloc[ph_bar - 1]) and
                      float(high.iloc[ph_bar]) > float(high.iloc[ph_bar + 1]))

        rsi_ok_long  = r < self.RSI_BUY_MAX
        rsi_ok_short = r > self.RSI_SELL_MIN

        go_long  = pivot_low  and rsi_ok_long
        go_short = pivot_high and rsi_ok_short

        signals = []

        if go_long:
            sl  = entry - atr_v * self.SL_ATR
            tp  = entry + atr_v * self.TP_ATR
            signals.append({
                "type":        "buy",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": entry + atr_v * 0.3,
                                "low":  float(low.iloc[pl_bar])},
                "pattern_key": "nq_pivot_long",
                "strategy":    self.name,
                "notes":       (f"Pivot low confirmed, RSI={r:.1f}, "
                                f"ATR={atr_v:.2f}, daily_pnl={self._daily_pnl:.0f}"),
            })

        if go_short:
            sl  = entry + atr_v * self.SL_ATR
            tp  = entry - atr_v * self.TP_ATR
            signals.append({
                "type":        "sell",
                "entry_price": entry,
                "sl_price":    sl,
                "tp_price":    tp,
                "quality":     7.0,
                "zone":        {"high": float(high.iloc[ph_bar]),
                                "low":  entry - atr_v * 0.3},
                "pattern_key": "nq_pivot_short",
                "strategy":    self.name,
                "notes":       (f"Pivot high confirmed, RSI={r:.1f}, "
                                f"ATR={atr_v:.2f}, daily_pnl={self._daily_pnl:.0f}"),
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
