"""
core/exchange/kucoin_exchange.py — KuCoin adapter.
KuCoin Futures uses 'kucoinfutures'; spot uses 'kucoin'.
Both require passphrase (set via OKX_PASSPHRASE or KUCOIN_PASSPHRASE in .env).
Symbol format futures: 'BTC/USDT:USDT'
Symbol format spot:    'BTC/USDT'
"""

from core.exchange.ccxt_exchange import CCXTExchange


class KuCoinFuturesExchange(CCXTExchange):
    """KuCoin Futures (USDT-margined perpetuals)."""

    name             = "KuCoinFutures"
    exchange_id      = "kucoinfutures"
    market_type      = "swap"
    default_settle   = "USDT"
    supports_futures = True
    supports_spot    = False


class KuCoinSpotExchange(CCXTExchange):
    """KuCoin Spot market."""

    name             = "KuCoin"
    exchange_id      = "kucoin"
    market_type      = "spot"
    default_settle   = "USDT"
    supports_futures = False
    supports_spot    = True


# Default alias: futures
KuCoinExchange = KuCoinFuturesExchange
