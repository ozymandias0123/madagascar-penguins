"""
core/exchange/kraken_exchange.py — Kraken adapter.
Kraken Futures uses a separate ccxt id: 'krakenfutures'.
Spot uses standard 'kraken'.
Symbol format spot: 'BTC/USD', 'ETH/USD'
Symbol format futures: 'BTC/USD:USD'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class KrakenFuturesExchange(CCXTExchange):
    """Kraken Futures (perpetuals)."""

    name             = "KrakenFutures"
    exchange_id      = "krakenfutures"
    market_type      = "future"
    default_settle   = "USD"
    supports_futures = True
    supports_spot    = False


class KrakenSpotExchange(CCXTExchange):
    """Kraken Spot market."""

    name             = "Kraken"
    exchange_id      = "kraken"
    market_type      = "spot"
    default_settle   = "USD"
    supports_futures = False
    supports_spot    = True


# Default alias
KrakenExchange = KrakenFuturesExchange
