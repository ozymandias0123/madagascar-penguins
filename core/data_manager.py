"""
core/data_manager.py — MT5 data fetching + ICT indicator computation.
Ported from ozy.py / DataManager unchanged.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import talib

from config import Config
from core.market_structure import MarketStructureDetector
from utils.mt5_manager import timed_function


class DataManager:

    @timed_function
    def get_candles(self, symbol: str, timeframe: int, count: int,
                    start_date=None, end_date=None) -> Optional[pd.DataFrame]:
        try:
            effective_count = max(count, Config.MIN_CANDLES)

            if Config.BACKTEST_MODE and start_date and end_date:
                start = pd.Timestamp(start_date, tz='UTC')
                end   = (pd.Timestamp(end_date, tz='UTC') +
                         pd.Timedelta(hours=23, minutes=59, seconds=59))
                rates = mt5.copy_rates_range(symbol, timeframe, start, end)
            else:
                # MT5 must already be connected (initialized in engine.run())
                # Do NOT call mt5.initialize() here — it auto-launches MT5
                if mt5.account_info() is None:
                    logging.warning("[DataManager] MT5 not connected — skipping candle fetch")
                    return None
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, effective_count)

            if rates is None or len(rates) == 0:
                logging.error(f"[NO_DATA] {symbol}")
                return None

            df = pd.DataFrame(rates)
            required = ['open', 'high', 'low', 'close', 'time']
            if not all(c in df.columns for c in required):
                return None
            if (df['high'] < df['low']).any() or (df['close'] > df['high']).any():
                return None

            # Parse timestamps
            for unit in ['s', 'ms', 'us']:
                df['time'] = pd.to_datetime(df['time'], unit=unit,
                                            errors='coerce', utc=True)
                if df['time'].isna().sum() == 0:
                    break
            df = df.dropna(subset=['time'])
            if df.empty:
                return None

            df.set_index('time', inplace=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.DatetimeIndex(
                        pd.to_datetime(df.index, errors='coerce', utc=True), tz='UTC')
                    if df.index.isna().any():
                        return None
                except Exception:
                    return None

            df['symbol'] = symbol
            df = self._filter_trading_hours(df)
            df = self._add_ict_indicators(df)

            if len(df) < Config.MIN_CANDLES:
                return None
            return df

        except Exception as exc:
            logging.error(f"[DATA_ERROR] {symbol}: {exc}")
            return None

    # ── HTF helpers ───────────────────────────────────────────

    def get_h1_bias(self, symbol: str) -> str:
        try:
            h1_df = self.get_candles(symbol, mt5.TIMEFRAME_H1, 50)
            if h1_df is None or len(h1_df) < 21:
                return 'neutral'
            completed        = h1_df.iloc[:-1].copy()
            completed['ema20'] = completed['close'].ewm(span=20, adjust=False).mean()
            completed['ema50'] = (
                completed['close'].ewm(span=50, adjust=False).mean()
                if len(completed) >= 50 else completed['ema20']
            )
            p   = completed['close'].iloc[-1]
            e20 = completed['ema20'].iloc[-1]
            e50 = completed['ema50'].iloc[-1]
            if p > e20 and e20 > e50:
                return 'bullish'
            elif p < e20 and e20 < e50:
                return 'bearish'
            return 'neutral'
        except Exception as exc:
            logging.error(f"[HTF_BIAS_ERROR] {exc}")
            return 'neutral'

    def get_htf_structure(self, symbol: str) -> dict:
        try:
            htf_df = self.get_candles(symbol, Config.HTF_STRUCTURE, 100)
            if htf_df is None or len(htf_df) < 20:
                return {'structure': 'neutral', 'bos': False, 'choch': False}
            structure = MarketStructureDetector.detect_market_structure(htf_df)
            return {
                'structure': structure,
                'bos':   'bos'   in structure,
                'choch': 'choch' in structure,
                'timeframe': 'H1'
            }
        except Exception as exc:
            logging.error(f"[HTF_STRUCTURE_ERROR] {exc}")
            return {'structure': 'neutral', 'bos': False, 'choch': False}

    # ── Private helpers ───────────────────────────────────────

    def _filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(df.index, pd.DatetimeIndex) or not hasattr(df.index, 'hour'):
                return df
            mask = (df.index.hour.between(7, 22)) & (
                ((df.index.hour >= 7)  & (df.index.hour < 16)) |
                ((df.index.hour >= 13) & (df.index.hour < 22))
            )
            return df[mask]
        except Exception:
            return df

    def _add_ict_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df['atr'] = self._calculate_atr(df, Config.ATR_PERIOD)
        df['fvg'] = self._calculate_fvg(df)
        df['rsi'] = self._calculate_rsi(df, 14)
        df['ema_20']  = df['close'].ewm(span=20,  adjust=False).mean()
        df['ema_50']  = df['close'].ewm(span=50,  adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        try:
            df['adx_indicator'] = talib.ADX(
                df['high'].values, df['low'].values,
                df['close'].values, timeperiod=14
            )
        except Exception:
            df['adx_indicator'] = 20.0
        return df.bfill().ffill()

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift(1)).abs()
        lc = (df['low']  - df['close'].shift(1)).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period, min_periods=1).mean()

    @staticmethod
    def _calculate_fvg(df: pd.DataFrame) -> pd.Series:
        bullish = (df['low'].shift(3) > df['high'].shift(1)) & \
                  (df['close'].shift(2) < df['low'].shift(3))
        bearish = (df['high'].shift(3) < df['low'].shift(1)) & \
                  (df['close'].shift(2) > df['high'].shift(3))
        fvg = pd.Series(index=df.index, dtype='object')
        fvg[bullish] = 'bullish'
        fvg[bearish] = 'bearish'
        return fvg

    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int) -> pd.Series:
        delta = df['close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
        loss  = -delta.where(delta < 0, 0).rolling(period, min_periods=1).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-10)))
