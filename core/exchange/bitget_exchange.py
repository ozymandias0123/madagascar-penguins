"""
core/exchange/bitget_exchange.py — Bitget adapter.
Popular for copy-trading. Requires passphrase.
Symbol format futures: 'BTC/USDT:USDT'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class BitgetExchange(CCXTExchange):
    """Bitget Perpetual Futures (USDT-margined)."""

    name             = "Bitget"
    exchange_id      = "bitget"
    market_type      = "swap"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = True


class BitgetSpotExchange(CCXTExchange):
    """Bitget Spot market."""

    name             = "BitgetSpot"
    exchange_id      = "bitget"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True
