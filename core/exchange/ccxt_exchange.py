"""
core/exchange/ccxt_exchange.py — Generic CCXT adapter.

One class covers every ccxt-compatible exchange (100 + exchanges).
Thin subclasses (binance_exchange.py, bybit_exchange.py, …) only set
class-level attributes; all heavy lifting happens here.

Dependencies:
    pip install ccxt>=4.3.0
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from core.exchange.base_exchange import BaseExchange, OrderResult, PositionInfo

logger = logging.getLogger(__name__)


class CCXTExchange(BaseExchange):
    """
    Generic adapter for any ccxt-compatible exchange.

    Subclass attributes to override
    ────────────────────────────────────────────────────────
    exchange_id    str   ccxt exchange id, e.g. 'binance'
    market_type    str   'future' | 'spot' | 'swap'
    default_settle str   settlement currency, e.g. 'USDT'
    supports_futures bool
    supports_spot    bool
    """

    exchange_id:     str  = ""           # must be set by subclass
    market_type:     str  = "future"     # 'spot' | 'future' | 'swap'
    default_settle:  str  = "USDT"
    supports_futures: bool = True
    supports_spot:    bool = True

    def __init__(
        self,
        api_key:    str  = "",
        api_secret: str  = "",
        passphrase: str  = "",          # OKX, KuCoin, Bitget need this
        testnet:    bool = False,
    ):
        self._api_key    = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._testnet    = testnet
        self._ex         = None     # ccxt exchange instance
        self._connected  = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            import ccxt
            cls = getattr(ccxt, self.exchange_id, None)
            if cls is None:
                logger.error(f"[{self.name}] ccxt has no exchange '{self.exchange_id}'")
                return False

            params: Dict = {
                'apiKey':  self._api_key,
                'secret':  self._api_secret,
                'options': {'defaultType': self.market_type},
            }
            if self._passphrase:
                params['password'] = self._passphrase
            if self._testnet:
                params['sandbox'] = True

            self._ex = cls(params)
            self._ex.load_markets()
            self._connected = True
            logger.info(f"[{self.name}] ✅ Connected ({self.exchange_id}/{self.market_type})")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] connect failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._ex        = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ex is not None

    # ── Account ───────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        try:
            bal = self._ex.fetch_balance()
            for cur in [self.default_settle, 'USDT', 'USD', 'BUSD', 'USDC']:
                v = bal.get(cur, {}).get('free', 0)
                if v and float(v) > 0:
                    return float(v)
            # futures: look in 'info' / total
            total = bal.get('total', {})
            if total:
                return max((float(v) for v in total.values() if v), default=0.0)
            return 0.0
        except Exception as e:
            logger.error(f"[{self.name}] get_balance: {e}")
            return 0.0

    def get_account_info(self) -> Dict:
        try:
            bal     = self._ex.fetch_balance()
            balance = self.get_balance()
            equity  = balance   # approximate; some exchanges expose unrealised PnL separately
            try:
                positions = self._ex.fetch_positions()
                upnl = sum(float(p.get('unrealizedPnl', 0) or 0) for p in positions)
                equity = balance + upnl
            except Exception:
                pass
            return {
                'exchange':    self.exchange_id,
                'balance':     balance,
                'equity':      equity,
                'margin_free': balance,
                'leverage':    10,           # default; subclass can override
                'currency':    self.default_settle,
            }
        except Exception as e:
            return {'exchange': self.exchange_id, 'balance': 0.0, 'error': str(e)}

    # ── Market data ───────────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        try:
            t = self._ex.fetch_ticker(symbol)
            return float(t.get('last') or t.get('bid') or 0)
        except Exception as e:
            logger.error(f"[{self.name}] get_price({symbol}): {e}")
            return 0.0

    def get_spread(self, symbol: str) -> float:
        try:
            t   = self._ex.fetch_ticker(symbol)
            bid = float(t.get('bid') or 0)
            ask = float(t.get('ask') or 0)
            return ask - bid if ask > bid else 0.0
        except Exception:
            return 0.0

    def get_candles(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        _EMPTY = pd.DataFrame(columns=['time','open','high','low','close','volume'])
        try:
            ohlcv = self._ex.fetch_ohlcv(symbol, timeframe, limit=count)
            if not ohlcv:
                return _EMPTY
            df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            return df.sort_values('time').reset_index(drop=True)
        except Exception as e:
            logger.error(f"[{self.name}] get_candles({symbol},{timeframe}): {e}")
            return _EMPTY

    # ── Trading ───────────────────────────────────────────────────────────────

    def place_order(self, symbol: str, order_type: str, lot: float,
                    sl_price: float, tp_price: float, comment: str = "") -> OrderResult:
        try:
            side  = order_type.lower()
            order = self._ex.create_order(
                symbol=symbol, type='market', side=side, amount=lot
            )
            ticket = int(order['id']) if str(order.get('id', '')).isdigit() else hash(order.get('id','')) & 0xFFFFFF
            filled = float(order.get('average') or order.get('price') or 0)

            # Attempt SL / TP as separate orders (not all exchanges support it)
            self._set_sl_tp(symbol, side, lot, sl_price, tp_price)

            logger.info(f"[{self.name}] ✅ {side.upper()} {lot} {symbol} @ {filled:.4f}")
            return OrderResult(
                success=True, ticket=ticket, symbol=symbol,
                order_type=side, lot=lot, entry_price=filled,
                sl_price=sl_price, tp_price=tp_price, raw=order,
            )
        except Exception as e:
            logger.error(f"[{self.name}] place_order: {e}")
            return OrderResult(success=False, error=str(e))

    def _set_sl_tp(self, symbol: str, entry_side: str,
                   lot: float, sl: float, tp: float) -> None:
        """Best-effort SL/TP placement. Silently skipped if unsupported."""
        close_side = 'sell' if entry_side == 'buy' else 'buy'
        try:
            if sl > 0:
                self._ex.create_order(
                    symbol=symbol, type='stop_market', side=close_side, amount=lot,
                    params={'stopPrice': sl, 'reduceOnly': True},
                )
        except Exception as e:
            logger.debug(f"[{self.name}] SL order skipped: {e}")
        try:
            if tp > 0:
                self._ex.create_order(
                    symbol=symbol, type='take_profit_market', side=close_side, amount=lot,
                    params={'stopPrice': tp, 'reduceOnly': True},
                )
        except Exception as e:
            logger.debug(f"[{self.name}] TP order skipped: {e}")

    def close_position(self, ticket: int, symbol: str = "") -> bool:
        try:
            syms = [symbol] if symbol else None
            positions = self._ex.fetch_positions(syms)
            closed = False
            for pos in positions:
                contracts = abs(float(pos.get('contracts') or 0))
                if contracts < 1e-9:
                    continue
                close_side = 'sell' if (pos.get('side') or '').lower() == 'long' else 'buy'
                self._ex.create_order(
                    symbol=pos['symbol'], type='market',
                    side=close_side, amount=contracts,
                    params={'reduceOnly': True},
                )
                closed = True
            return closed
        except Exception as e:
            logger.error(f"[{self.name}] close_position: {e}")
            return False

    def get_open_positions(self) -> List[PositionInfo]:
        try:
            raw = self._ex.fetch_positions()
            result = []
            for p in raw:
                contracts = abs(float(p.get('contracts') or 0))
                if contracts < 1e-9:
                    continue
                result.append(PositionInfo(
                    ticket=hash(str(p.get('id', p.get('symbol', '')))),
                    symbol=p.get('symbol', ''),
                    order_type='buy' if (p.get('side') or '').lower() == 'long' else 'sell',
                    volume=contracts,
                    price=float(p.get('entryPrice') or 0),
                    current=float(p.get('markPrice') or 0),
                    profit=float(p.get('unrealizedPnl') or 0),
                    sl=float(p.get('stopLossPrice') or 0),
                    tp=float(p.get('takeProfitPrice') or 0),
                ))
            return result
        except Exception as e:
            logger.error(f"[{self.name}] get_open_positions: {e}")
            return []

    # ── Risk / Lot size ───────────────────────────────────────────────────────

    def calculate_lot(self, balance: float, risk_pct: float,
                      sl_distance: float, symbol: str) -> float:
        if sl_distance <= 0:
            return 0.001
        risk_amount = balance * risk_pct
        lot         = risk_amount / sl_distance
        try:
            market    = self._ex.market(symbol)
            limits    = market.get('limits', {}).get('amount', {})
            min_lot   = float(limits.get('min') or 0.001)
            precision = market.get('precision', {}).get('amount', 3)
            lot       = max(round(lot, int(precision)), min_lot)
        except Exception:
            lot = max(round(lot, 3), 0.001)
        return lot
