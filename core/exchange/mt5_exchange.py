"""
core/exchange/mt5_exchange.py — MetaTrader 5 exchange adapter.

Wraps the MetaTrader5 Python library behind the BaseExchange interface.
Existing engine.py code continues to work unchanged; this adapter is used
when ACTIVE_EXCHANGE = 'mt5' in config.
"""

import logging
from typing import Dict, List

import pandas as pd

from config import Config
from core.exchange.base_exchange import BaseExchange, OrderResult, PositionInfo

logger = logging.getLogger(__name__)


class MT5Exchange(BaseExchange):
    """MetaTrader 5 adapter."""

    name             = "MT5"
    supports_futures = True
    supports_spot    = False

    def __init__(self, login: str = "", password: str = "", server: str = ""):
        self._login    = login    or Config.LOGIN
        self._password = password or Config.PASSWORD
        self._server   = server   or Config.SERVER
        self._connected = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.error("[MT5] initialize() failed")
                return False
            if not mt5.login(int(self._login), self._password, self._server):
                err = mt5.last_error()
                logger.error(f"[MT5] login failed: {err}")
                mt5.shutdown()
                return False
            self._connected = True
            logger.info(f"[MT5] ✅ Connected: login={self._login} server={self._server}")
            return True
        except Exception as e:
            logger.error(f"[MT5] connect error: {e}")
            return False

    def disconnect(self) -> None:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            return float(info.balance) if info else Config.INITIAL_BALANCE
        except Exception:
            return Config.INITIAL_BALANCE

    def get_account_info(self) -> Dict:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                return {
                    'exchange':    'mt5',
                    'balance':     float(info.balance),
                    'equity':      float(info.equity),
                    'margin_free': float(info.margin_free),
                    'leverage':    int(info.leverage),
                    'currency':    info.currency,
                }
        except Exception as e:
            logger.error(f"[MT5] get_account_info error: {e}")
        return {'exchange': 'mt5', 'balance': self.get_balance()}

    # ── Market data ───────────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            return float(tick.last) if tick else 0.0
        except Exception:
            return 0.0

    def get_spread(self, symbol: str) -> float:
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return float(tick.ask - tick.bid)
        except Exception:
            pass
        return 0.0

    def get_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        try:
            import MetaTrader5 as mt5
            mt5_tf = self.ccxt_tf_to_mt5(timeframe)
            rates  = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates is None or len(rates) == 0:
                return pd.DataFrame(columns=['time','open','high','low','close','volume'])
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'tick_volume': 'volume'})[
                ['time','open','high','low','close','volume']
            ]
            return df.reset_index(drop=True)
        except Exception as e:
            logger.error(f"[MT5] get_candles error: {e}")
            return pd.DataFrame(columns=['time','open','high','low','close','volume'])

    # ── Trading ───────────────────────────────────────────────────────────────

    def place_order(self, symbol: str, order_type: str, lot: float,
                    sl_price: float, tp_price: float, comment: str = "") -> OrderResult:
        try:
            import MetaTrader5 as mt5
            side  = order_type.lower()
            mt5_type = mt5.ORDER_TYPE_BUY if side == 'buy' else mt5.ORDER_TYPE_SELL
            tick  = mt5.symbol_info_tick(symbol)
            price = tick.ask if side == 'buy' else tick.bid

            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    symbol,
                "volume":    float(lot),
                "type":      mt5_type,
                "price":     price,
                "sl":        float(sl_price),
                "tp":        float(tp_price),
                "deviation": 20,
                "magic":     20250101,
                "comment":   comment or "PenguinSquad",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MT5] ✅ Order: {side} {lot} {symbol} ticket={result.order}")
                return OrderResult(
                    success=True, ticket=result.order, symbol=symbol,
                    order_type=side, lot=lot, entry_price=result.price,
                    sl_price=sl_price, tp_price=tp_price,
                )
            err = result.comment if result else "unknown"
            logger.error(f"[MT5] Order failed: {err}")
            return OrderResult(success=False, error=err)
        except Exception as e:
            logger.error(f"[MT5] place_order exception: {e}")
            return OrderResult(success=False, error=str(e))

    def close_position(self, ticket: int, symbol: str = "") -> bool:
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            pos  = positions[0]
            side = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            price = tick.bid if pos.type == 0 else tick.ask
            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    pos.symbol,
                "volume":    pos.volume,
                "type":      side,
                "position":  ticket,
                "price":     price,
                "deviation": 20,
                "magic":     20250101,
                "comment":   "close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        except Exception as e:
            logger.error(f"[MT5] close_position error: {e}")
            return False

    def get_open_positions(self) -> List[PositionInfo]:
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get()
            if not positions:
                return []
            result = []
            for p in positions:
                tick = mt5.symbol_info_tick(p.symbol)
                current = (tick.bid + tick.ask) / 2 if tick else p.price_open
                result.append(PositionInfo(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    order_type='buy' if p.type == 0 else 'sell',
                    volume=p.volume,
                    price=p.price_open,
                    current=current,
                    profit=p.profit,
                    sl=p.sl,
                    tp=p.tp,
                    comment=p.comment,
                ))
            return result
        except Exception as e:
            logger.error(f"[MT5] get_open_positions error: {e}")
            return []

    # ── Risk ──────────────────────────────────────────────────────────────────

    def calculate_lot(self, balance: float, risk_pct: float,
                      sl_distance: float, symbol: str) -> float:
        if sl_distance <= 0:
            return Config.MAX_LOT_SIZE * 0.1
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"symbol_info({symbol}) returned None")
            risk_amount  = balance * risk_pct
            tick_value   = info.trade_tick_value
            tick_size    = info.trade_tick_size
            lot_step     = info.volume_step
            min_lot      = info.volume_min

            ticks_in_sl  = sl_distance / tick_size if tick_size > 0 else sl_distance
            lot          = risk_amount / (ticks_in_sl * tick_value) if tick_value > 0 else 0.01
            lot          = max(round(lot / lot_step) * lot_step, min_lot)
            lot          = min(lot, Config.MAX_LOT_SIZE)
            return round(lot, 2)
        except Exception as e:
            logger.warning(f"[MT5] calculate_lot fallback: {e}")
            return min(round(balance * risk_pct / (sl_distance * 100 + 1), 2), Config.MAX_LOT_SIZE)
