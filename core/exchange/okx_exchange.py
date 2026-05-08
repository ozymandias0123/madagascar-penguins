"""
core/exchange/okx_exchange.py — OKX adapter (Swap / Futures).
OKX requires an API passphrase in addition to key + secret.
Symbol format: 'BTC/USDT:USDT', 'ETH/USDT:USDT'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class OKXExchange(CCXTExchange):
    """OKX Perpetual Swap (USDT-margined)."""

    name             = "OKX"
    exchange_id      = "okx"
    market_type      = "swap"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = True


class OKXSpotExchange(CCXTExchange):
    """OKX Spot market."""

    name             = "OKXSpot"
    exchange_id      = "okx"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True
