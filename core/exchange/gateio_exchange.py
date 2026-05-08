"""
core/exchange/gateio_exchange.py — Gate.io adapter.
Gate.io supports both futures and spot via separate ccxt calls.
Symbol format futures: 'BTC/USDT:USDT'
Symbol format spot:    'BTC/USDT'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class GateIOFuturesExchange(CCXTExchange):
    """Gate.io Perpetual Futures (USDT-settled)."""

    name             = "GateIOFutures"
    exchange_id      = "gate"
    market_type      = "future"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = True


class GateIOSpotExchange(CCXTExchange):
    """Gate.io Spot market."""

    name             = "GateIO"
    exchange_id      = "gate"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True


# Default alias: futures
GateIOExchange = GateIOFuturesExchange
