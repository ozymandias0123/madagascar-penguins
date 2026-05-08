"""
core/exchange/__init__.py — Exchange factory & registry.

Usage
─────
    from core.exchange import get_exchange, list_exchanges

    ex = get_exchange('binance',
                      api_key='...', api_secret='...', testnet=False)
    ex.connect()
    print(ex.get_balance())

Supported exchanges
───────────────────
    mt5        — MetaTrader 5 (forex + indices)
    binance    — Binance USD-M Futures
    bybit      — Bybit Linear Perpetuals
    okx        — OKX Swap (requires passphrase)
    kraken     — Kraken Futures
    kucoin     — KuCoin Futures (requires passphrase)
    gateio     — Gate.io Futures
    bitget     — Bitget Swap (requires passphrase)
    mexc       — MEXC Swap

All subclasses also expose a *Spot variant (e.g. 'binance_spot').
"""

from typing import Dict, Type

from core.exchange.base_exchange    import BaseExchange, OrderResult, PositionInfo
from core.exchange.mt5_exchange     import MT5Exchange
from core.exchange.binance_exchange import BinanceExchange, BinanceSpotExchange
from core.exchange.bybit_exchange   import BybitExchange,   BybitSpotExchange
from core.exchange.okx_exchange     import OKXExchange,     OKXSpotExchange
from core.exchange.kraken_exchange  import KrakenExchange,  KrakenSpotExchange
from core.exchange.kucoin_exchange  import KuCoinExchange,  KuCoinSpotExchange
from core.exchange.gateio_exchange  import GateIOExchange,  GateIOSpotExchange
from core.exchange.bitget_exchange  import BitgetExchange,  BitgetSpotExchange
from core.exchange.mexc_exchange    import MEXCExchange,    MEXCSpotExchange

# ── Registry ──────────────────────────────────────────────────────────────────

EXCHANGE_REGISTRY: Dict[str, Type[BaseExchange]] = {
    # MetaTrader
    'mt5':          MT5Exchange,
    # Binance
    'binance':      BinanceExchange,
    'binance_spot': BinanceSpotExchange,
    # Bybit
    'bybit':        BybitExchange,
    'bybit_spot':   BybitSpotExchange,
    # OKX
    'okx':          OKXExchange,
    'okx_spot':     OKXSpotExchange,
    # Kraken
    'kraken':       KrakenExchange,
    'kraken_spot':  KrakenSpotExchange,
    # KuCoin
    'kucoin':       KuCoinExchange,
    'kucoin_spot':  KuCoinSpotExchange,
    # Gate.io
    'gateio':       GateIOExchange,
    'gateio_spot':  GateIOSpotExchange,
    # Bitget
    'bitget':       BitgetExchange,
    'bitget_spot':  BitgetSpotExchange,
    # MEXC
    'mexc':         MEXCExchange,
    'mexc_spot':    MEXCSpotExchange,
}


def get_exchange(name: str, **kwargs) -> BaseExchange:
    """
    Instantiate and return an exchange adapter by name.

    Parameters
    ----------
    name       : one of EXCHANGE_REGISTRY keys (case-insensitive)
    **kwargs   : passed to the adapter constructor
                 MT5      → login, password, server
                 Crypto   → api_key, api_secret, passphrase, testnet
    """
    key = name.lower().strip()
    cls = EXCHANGE_REGISTRY.get(key)
    if cls is None:
        available = ', '.join(sorted(EXCHANGE_REGISTRY))
        raise ValueError(
            f"Unknown exchange '{name}'. Available: {available}"
        )
    return cls(**kwargs)


def list_exchanges() -> list:
    """Return sorted list of all supported exchange names."""
    return sorted(EXCHANGE_REGISTRY.keys())


def get_exchange_from_config() -> BaseExchange:
    """
    Build the active exchange from Config settings.
    Falls back to MT5 if ACTIVE_EXCHANGE is not set.
    """
    from config import Config

    name = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').lower()

    if name == 'mt5':
        return get_exchange(
            'mt5',
            login=Config.LOGIN,
            password=Config.PASSWORD,
            server=Config.SERVER,
        )

    # Crypto exchange — look up API keys from config
    key_map = {
        'binance':   ('BINANCE_API_KEY',   'BINANCE_API_SECRET',   ''),
        'binance_spot': ('BINANCE_API_KEY','BINANCE_API_SECRET',   ''),
        'bybit':     ('BYBIT_API_KEY',     'BYBIT_API_SECRET',     ''),
        'bybit_spot':('BYBIT_API_KEY',     'BYBIT_API_SECRET',     ''),
        'okx':       ('OKX_API_KEY',       'OKX_API_SECRET',       'OKX_PASSPHRASE'),
        'okx_spot':  ('OKX_API_KEY',       'OKX_API_SECRET',       'OKX_PASSPHRASE'),
        'kraken':    ('KRAKEN_API_KEY',     'KRAKEN_API_SECRET',    ''),
        'kraken_spot':('KRAKEN_API_KEY',   'KRAKEN_API_SECRET',    ''),
        'kucoin':    ('KUCOIN_API_KEY',     'KUCOIN_API_SECRET',    'KUCOIN_PASSPHRASE'),
        'kucoin_spot':('KUCOIN_API_KEY',   'KUCOIN_API_SECRET',    'KUCOIN_PASSPHRASE'),
        'gateio':    ('GATEIO_API_KEY',     'GATEIO_API_SECRET',    ''),
        'gateio_spot':('GATEIO_API_KEY',   'GATEIO_API_SECRET',    ''),
        'bitget':    ('BITGET_API_KEY',     'BITGET_API_SECRET',    'BITGET_PASSPHRASE'),
        'bitget_spot':('BITGET_API_KEY',   'BITGET_API_SECRET',    'BITGET_PASSPHRASE'),
        'mexc':      ('MEXC_API_KEY',       'MEXC_API_SECRET',      ''),
        'mexc_spot': ('MEXC_API_KEY',       'MEXC_API_SECRET',      ''),
    }

    key_attr, secret_attr, pass_attr = key_map.get(name, ('', '', ''))
    return get_exchange(
        name,
        api_key    = getattr(Config, key_attr,   ''),
        api_secret = getattr(Config, secret_attr,''),
        passphrase = getattr(Config, pass_attr,  '') if pass_attr else '',
        testnet    = getattr(Config, 'EXCHANGE_TESTNET', False),
    )


__all__ = [
    'BaseExchange', 'OrderResult', 'PositionInfo',
    'MT5Exchange',
    'BinanceExchange', 'BinanceSpotExchange',
    'BybitExchange',   'BybitSpotExchange',
    'OKXExchange',     'OKXSpotExchange',
    'KrakenExchange',  'KrakenSpotExchange',
    'KuCoinExchange',  'KuCoinSpotExchange',
    'GateIOExchange',  'GateIOSpotExchange',
    'BitgetExchange',  'BitgetSpotExchange',
    'MEXCExchange',    'MEXCSpotExchange',
    'EXCHANGE_REGISTRY',
    'get_exchange',
    'list_exchanges',
    'get_exchange_from_config',
]
