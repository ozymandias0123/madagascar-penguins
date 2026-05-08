"""
core/exchange/mexc_exchange.py — MEXC Global adapter.
Known for wide altcoin selection and low fees.
Symbol format futures: 'BTC/USDT:USDT'
Symbol format spot:    'BTC/USDT'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class MEXCExchange(CCXTExchange):
    """MEXC Futures (USDT-margined perpetuals)."""

    name             = "MEXC"
    exchange_id      = "mexc"
    market_type      = "swap"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = True


class MEXCSpotExchange(CCXTExchange):
    """MEXC Spot market."""

    name             = "MEXCSpot"
    exchange_id      = "mexc"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True
