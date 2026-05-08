"""
config.py — Centralised configuration for the multi-agent trading system.
All classes import Config from here; nothing reads from ozy.py directly.
"""

import os
import logging
from dotenv import load_dotenv

# Load from safe location first (~/.penguin_squad/.env) — outside OneDrive/cloud sync
_SAFE_ENV = os.path.join(os.path.expanduser("~"), ".penguin_squad", ".env")
_LOCAL_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

if os.path.exists(_SAFE_ENV):
    load_dotenv(_SAFE_ENV, override=False)
load_dotenv(_LOCAL_ENV, override=False)   # fallback / legacy


class Config:
    # ── Credentials ───────────────────────────────────────────
    LOGIN    = os.getenv('MT5_LOGIN', '')
    PASSWORD = os.getenv('MT5_PASSWORD', '')
    SERVER   = os.getenv('MT5_SERVER', 'Exness-MT5Trial15')

    # ── Mode ──────────────────────────────────────────────────
    MODE          = os.getenv('BOT_MODE', 'demo').strip().lower()
    BACKTEST_MODE = MODE == 'backtest'
    PAPER_TRADING = MODE == 'demo'
    LIVE_MODE     = MODE == 'live'

    # ── Symbol ────────────────────────────────────────────────
    SYMBOLS = ['USTECm']

    # ── Timeframes (MT5 constants — no import needed at startup) ─
    # M15=15, H1=16385, H4=16388  (mt5.TIMEFRAME_* values)
    TIMEFRAME     = 15      # mt5.TIMEFRAME_M15
    HTF_STRUCTURE = 16385   # mt5.TIMEFRAME_H1
    HTF_BIAS      = 16388   # mt5.TIMEFRAME_H4

    # ── Risk ──────────────────────────────────────────────────
    RISK_PERCENT             = 0.005
    RR_RATIO                 = 3.0
    TRAILING_STOP_ATR        = 1.5
    SL_ATR_MULTIPLIER        = 1.5
    MIN_SL_POINTS            = 12
    MAX_SL_PRICE             = 50.0
    MAX_TP_PRICE             = 150.0
    MIN_SL_PRICE             = 15.0
    HIGH_VOL_SL_MULTIPLIER   = 2.0
    HIGH_VOL_RR_RATIO        = 4.0
    ORDER_BLOCK_SL           = True

    # ── Data / Candles ────────────────────────────────────────
    MIN_CANDLES   = 100
    ATR_PERIOD    = 14

    # ── Retry / Connection ────────────────────────────────────
    MAX_RETRIES   = 5
    RETRY_DELAY   = 5

    # ── Sessions ──────────────────────────────────────────────
    TRADING_SESSIONS  = [('london', 8, 12), ('new_york', 13, 22)]
    SESSION_STRATEGIES = {'london': ['ict'], 'new_york': ['ict']}
    SESSION_MAX_TRADES = {'london': 9999, 'new_york': 9999}

    # ── Trading Limits ────────────────────────────────────────
    MAX_DAILY_TRADES         = 999
    MAX_TRADE_DURATION_MINUTES = 480
    MAX_SPREAD               = 200.0
    MAX_LOT_SIZE             = 0.1
    MIN_BALANCE              = 1.0
    MAX_DRAWDOWN             = 1.0
    COMMISSION_PER_LOT       = 7.0
    MIN_TRADE_COST           = 0.01

    # ── Balance / Account ─────────────────────────────────────
    INITIAL_BALANCE  = 186.0
    BALANCE_TIER     = '186'

    # ── Circuit Breaker ───────────────────────────────────────
    CIRCUIT_BREAKER_ENABLED       = False
    LIVE_MAX_DRAWDOWN             = 0.99
    LIVE_DAILY_LOSS_LIMIT         = 0.99
    LIVE_CONSECUTIVE_LOSSES       = 999
    LIVE_MIN_BALANCE_PERCENT      = 0.01
    CIRCUIT_BREAKER_COOLDOWN_HOURS = 0

    # ── Backtesting ───────────────────────────────────────────
    BACKTEST_START_DATE    = '2024-01-01'
    BACKTEST_END_DATE      = '2025-12-19'
    PER_BACKTEST_MAX_TRADES = 999999
    PER_DEMO_MAX_TRADES    = 999999
    VISUAL_DEMO            = True
    DEBUG_MODE             = False

    # ── ICT / Strategy ────────────────────────────────────────
    FIB_LEVELS                = [0.5, 0.618, 0.705, 0.786]
    FVG_SIZE_THRESHOLD        = 0.00010
    FIRST_FVG_ONLY            = False
    JUDAS_SWING_ENABLED       = True
    JUDAS_SWING_BONUS         = 0.8

    # ── Signal Quality ────────────────────────────────────────
    MIN_QUALITY         = 2.0
    MIN_QUALITY_BEARISH = 2.3
    MIN_SB_TARGET_POINTS = 20

    # ── Silver Bullet ─────────────────────────────────────────
    SILVER_BULLET_MODE    = 'boost'
    SILVER_BULLET_WINDOWS = {
        'london': (8, 9),
        'ny_am':  (15, 16),
        'ny_pm':  (19, 20)
    }
    SILVER_BULLET_BONUS = 1.0

    # ── Regime Detection ──────────────────────────────────────
    REGIME_DETECTION_ENABLED      = True
    ADX_TRENDING_THRESHOLD        = 25
    ATR_HIGH_VOLATILITY_MULTIPLIER = 2.0
    RANGING_ONLY_SB               = True

    # ── Volatility Filter ─────────────────────────────────────
    VOLATILITY_FILTER            = True
    ATR_VOLATILITY_THRESHOLD     = 3.0
    VOLUME_THRESHOLD_MULTIPLIER  = 0.7

    # ── ML ────────────────────────────────────────────────────
    ML_ENABLED              = True
    LEARNING_RATE           = 0.05
    CONFIDENCE_THRESHOLD    = 0.40
    USE_PROFIT_THRESHOLD    = True
    PROFIT_R_THRESHOLD      = 0.8
    MIN_TRAINING_SAMPLES    = 200
    EARLY_STOPPING_ROUNDS   = 50
    MIN_WIN_RATE            = 0.40
    OVERFITTING_THRESHOLD   = 0.15
    MIN_CV_F1_SCORE         = 0.42
    SAVE_INTERVAL_TRADES    = 5
    BACKUP_FILES_COUNT      = 3

    LOCKED_FEATURE_LIST = [
        'atr', 'volatility_ratio', 'bos_bullish', 'bos_bearish',
        'rsi', 'rsi_change', 'volume_ratio', 'price_momentum',
        'ema_distance', 'ema_slope', 'is_silver_bullet',
        'hour_of_day', 'candle_body_ratio', 'ote_distance',
        'adx', 'atr_trend', 'prev_day_direction',
        'price_vs_ema50', 'price_vs_ema200', 'session_phase',
    ]  # 20 features total

    # ── ML Health Monitor ─────────────────────────────────────
    ML_HEALTH_CHECK_ENABLED   = True
    ML_MIN_WIN_RATE_THRESHOLD = 0.30
    ML_LOOKBACK_TRADES        = 20
    ML_FREEZE_AFTER_BAD_STREAK = 10
    ML_UNFREEZE_AFTER_HOURS   = 4

    # ── Pattern Tracker ───────────────────────────────────────
    PERFORMANCE_TRACKING_ENABLED = True
    PATTERN_UPDATE_INTERVAL      = 25
    PATTERN_ANALYSIS_WINDOW      = 100
    MIN_PATTERN_SAMPLES          = 5
    PATTERN_WEIGHT_DECAY         = 0.85
    PATTERN_WEIGHT_BOOST         = 1.2
    PERFORMANCE_ALERT_DRAWDOWN   = 1.0

    PATTERN_WEIGHTS = {
        'sb+bos+judas': 1.3,    'sb+bos+htf_aligned': 1.2,
        'sb+fvg+inducement': 1.1, 'sb+bos': 1.0, 'sb+fvg': 1.0,
        'bos+htf_aligned+trending': 1.1, 'bos+judas': 1.0,
        'bos+inducement': 0.9,  'bos': 0.8,
        'choch+judas': 0.7,     'choch+htf_aligned': 0.6, 'choch': 0.5,
        'fvg': 0.6,             'high_vol+sb': 1.1,       'high_vol+bos': 0.9,
    }

    # ── Telegram (Rico) ───────────────────────────────────────
    # 1. t.me/BotFather → /newbot → TOKEN
    # 2. Add bot to group → get CHAT_ID via t.me/userinfobot
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID   = os.getenv('TELEGRAM_CHAT_ID',   '')
    TELEGRAM_ENABLED   = os.getenv('TELEGRAM_ENABLED', 'True').lower() in ['1', 'true', 'yes']

    # ─────────────────────────────────────────────────────────
    # Multi-Agent Orchestrator Configuration
    # ─────────────────────────────────────────────────────────
    ORCHESTRATOR_ENABLED  = os.getenv('ORCHESTRATOR_ENABLED', 'True').lower() in ['1', 'true', 'yes']
    AGENT_MIN_QUALITY     = float(os.getenv('AGENT_MIN_QUALITY', '3.0'))
    AGENT_MIN_CONFIDENCE  = int(os.getenv('AGENT_MIN_CONFIDENCE', '55'))

    # Individual LLM API Keys
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')   # Claude   → Kowalski
    OPENAI_API_KEY    = os.getenv('OPENAI_API_KEY', '')      # GPT-4o   → Skipper (if openai)
    GOOGLE_API_KEY    = os.getenv('GOOGLE_API_KEY', '')      # Gemini   → Skipper (if gemini) + Rico
    DEEPSEEK_API_KEY  = os.getenv('DEEPSEEK_API_KEY', '')    # DeepSeek → Private

    # Skipper provider: 'openai' (ChatGPT) or 'gemini'
    SKIPPER_PROVIDER  = os.getenv('SKIPPER_PROVIDER', 'openai').lower()

    # Free news sources
    FINNHUB_API_KEY   = os.getenv('FINNHUB_API_KEY', '')   # free: finnhub.io

    # ── Exchange selection ────────────────────────────────────────────────────
    # Options: mt5 | binance | bybit | okx | kraken | kucoin | gateio | bitget | mexc
    # Append _spot for spot market (e.g. 'binance_spot')
    ACTIVE_EXCHANGE  = os.getenv('ACTIVE_EXCHANGE', 'mt5').lower()
    EXCHANGE_TESTNET = os.getenv('EXCHANGE_TESTNET', 'False').lower() in ['1', 'true', 'yes']

    # ── Crypto Exchange API Keys ──────────────────────────────────────────────
    # Binance
    BINANCE_API_KEY    = os.getenv('BINANCE_API_KEY',    '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')

    # Bybit
    BYBIT_API_KEY      = os.getenv('BYBIT_API_KEY',      '')
    BYBIT_API_SECRET   = os.getenv('BYBIT_API_SECRET',   '')

    # OKX (requires passphrase)
    OKX_API_KEY        = os.getenv('OKX_API_KEY',        '')
    OKX_API_SECRET     = os.getenv('OKX_API_SECRET',     '')
    OKX_PASSPHRASE     = os.getenv('OKX_PASSPHRASE',     '')

    # Kraken
    KRAKEN_API_KEY     = os.getenv('KRAKEN_API_KEY',     '')
    KRAKEN_API_SECRET  = os.getenv('KRAKEN_API_SECRET',  '')

    # KuCoin (requires passphrase)
    KUCOIN_API_KEY     = os.getenv('KUCOIN_API_KEY',     '')
    KUCOIN_API_SECRET  = os.getenv('KUCOIN_API_SECRET',  '')
    KUCOIN_PASSPHRASE  = os.getenv('KUCOIN_PASSPHRASE',  '')

    # Gate.io
    GATEIO_API_KEY     = os.getenv('GATEIO_API_KEY',     '')
    GATEIO_API_SECRET  = os.getenv('GATEIO_API_SECRET',  '')

    # Bitget (requires passphrase)
    BITGET_API_KEY     = os.getenv('BITGET_API_KEY',     '')
    BITGET_API_SECRET  = os.getenv('BITGET_API_SECRET',  '')
    BITGET_PASSPHRASE  = os.getenv('BITGET_PASSPHRASE',  '')

    # MEXC
    MEXC_API_KEY       = os.getenv('MEXC_API_KEY',       '')
    MEXC_API_SECRET    = os.getenv('MEXC_API_SECRET',    '')

    # LLM model names
    OPENAI_MODEL   = 'gpt-4o'
    GEMINI_MODEL   = 'gemini-1.5-flash'
    CLAUDE_MODEL   = 'claude-sonnet-4-5'
    DEEPSEEK_MODEL = 'deepseek-chat'

    # Legacy single-agent fallback (original agents.py behaviour)
    AGENT_ENABLED = False   # kept for backward compat; superseded by ORCHESTRATOR_ENABLED

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def get_file_prefix() -> str:
        if Config.MODE == 'backtest':
            return 'backtest'
        elif Config.MODE == 'demo':
            return 'demo'
        return 'live'

    @staticmethod
    def get_mode_weight() -> float:
        if Config.MODE == 'backtest':
            return 1.0
        elif Config.MODE == 'demo':
            return 2.0
        return 3.0

    @staticmethod
    def validate():
        required = ['SYMBOLS', 'LOGIN', 'PASSWORD', 'SERVER',
                    'ATR_PERIOD', 'RR_RATIO', 'RISK_PERCENT',
                    'SL_ATR_MULTIPLIER', 'MAX_SPREAD']
        for attr in required:
            val = getattr(Config, attr, None)
            if val is None or val == '':
                raise ValueError(f"[CONFIG_ERROR] Missing or invalid: {attr}")
        if Config.MODE not in {'backtest', 'demo', 'live'}:
            raise ValueError("[CONFIG_ERROR] BOT_MODE must be: backtest | demo | live")
        if Config.TELEGRAM_ENABLED and not Config.TELEGRAM_BOT_TOKEN:
            logging.warning("[CONFIG] TELEGRAM_BOT_TOKEN not set — Telegram notifications disabled")
        logging.info("[CONFIG] ✅ Configuration validated")

    @staticmethod
    def update_credentials(login=None, password=None, server=None):
        if login    is not None: Config.LOGIN    = login
        if password is not None: Config.PASSWORD = password
        if server   is not None: Config.SERVER   = server
        logging.info(f"[CONFIG] Credentials updated: login={Config.LOGIN}, server={Config.SERVER}")
