"""
core/exchange/bybit_exchange.py — Bybit adapter (Linear Perpetuals).
Symbol format: 'BTC/USDT:USDT', 'ETH/USDT:USDT'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class BybitExchange(CCXTExchange):
    """Bybit Linear Perpetual Futures (USDT-margined)."""

    name             = "Bybit"
    exchange_id      = "bybit"
    market_type      = "linear"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = True


class BybitSpotExchange(CCXTExchange):
    """Bybit Spot market."""

    name             = "BybitSpot"
    exchange_id      = "bybit"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True
