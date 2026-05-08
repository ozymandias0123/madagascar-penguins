"""
agents/private.py — Private (Telegram Bot + Full Bot Control)

Visual menu structure
─────────────────────
/menu  (or /start after first run)
┌─────────────────────────────────────┐
│  🐧 Madagascar Penguins             │
│  ─────────────────────────────────  │
│  [🟡 Demo]      [🔴 Live]           │
│  [📡 Signals]   [🔬 Analysis]       │
│  [⏸ Pause]      [▶️ Resume]         │
│  [📊 Status]    [💰 Balance]        │
│  [🌐 Connect Exchange]              │
│  [⚙️ Settings]  [📈 Trades]         │
└─────────────────────────────────────┘

Settings submenu:
┌─────────────────────────────────────┐
│  ⚙️ SETTINGS                        │
│  [🌐 Exchange]  [📡 Symbol]         │
│  [⚡ Risk %]    [🔑 API Keys]       │
│  [◀️ Main Menu]                     │
└─────────────────────────────────────┘

Connect (exchange picker):
┌─────────────────────────────────────┐
│  🌐 CHOOSE EXCHANGE                 │
│  [1️⃣ MT5]     [2️⃣ Binance]        │
│  [3️⃣ Bybit]   [4️⃣ OKX]           │
│  [5️⃣ Kraken]  [6️⃣ KuCoin]         │
│  [7️⃣ Gate.io] [8️⃣ Bitget]         │
│  [9️⃣ MEXC]                         │
│  [◀️ Cancel]                        │
└─────────────────────────────────────┘

API Keys submenu:
┌─────────────────────────────────────┐
│  🔑 API KEYS                        │
│  [🎖 Skipper  OpenAI   ✅]          │
│  [🧠 Kowalski Claude   ✅]          │
│  [🃏 Rico     DeepSeek ❌]          │
│  [🌐 Gemini   Google   ✅]          │
│  [◀️ Settings]                      │
└─────────────────────────────────────┘
"""

import logging
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

from config import Config
from utils.telegram_notifier import (
    TelegramNotifier, TelegramCommandBot,
    Keyboard, get_notifier
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_EXCHANGE_LABELS = {
    'mt5':     '1️⃣ MT5',
    'binance': '2️⃣ Binance',
    'bybit':   '3️⃣ Bybit',
    'okx':     '4️⃣ OKX',
    'kraken':  '5️⃣ Kraken',
    'kucoin':  '6️⃣ KuCoin',
    'gateio':  '7️⃣ Gate.io',
    'bitget':  '8️⃣ Bitget',
    'mexc':    '9️⃣ MEXC',
}

_EXCHANGE_MAP = {
    '1': 'mt5',     'mt5':     'mt5',
    '2': 'binance', 'binance': 'binance',
    '3': 'bybit',   'bybit':   'bybit',
    '4': 'okx',     'okx':     'okx',
    '5': 'kraken',  'kraken':  'kraken',
    '6': 'kucoin',  'kucoin':  'kucoin',
    '7': 'gateio',  'gateio':  'gateio', 'gate.io': 'gateio',
    '8': 'bitget',  'bitget':  'bitget',
    '9': 'mexc',    'mexc':    'mexc',
}

_NEEDS_PASSPHRASE = {'okx', 'kucoin', 'bitget'}

_PENGUIN_KEYS = {
    'skipper':  ('OPENAI_API_KEY',    '🎖 Skipper',  'OpenAI'),
    'kowalski': ('ANTHROPIC_API_KEY', '🧠 Kowalski', 'Anthropic'),
    'rico':     ('DEEPSEEK_API_KEY',  '🃏 Rico',     'DeepSeek'),
    # 'gemini' removed — Google key is used internally; Private = Telegram (no key needed)
}

_MODE_ICONS = {
    'demo': '🟡', 'live': '🔴',
    'signals': '📡', 'analysis': '🔬', 'backtest': '🔵',
}


def _key_ok(attr: str) -> str:
    val = getattr(Config, attr, '')
    return '✅' if val and len(val) > 8 else '❌'


def _update_env(updates: dict) -> None:
    # Always write to the safe location (~/.penguin_squad/.env) — outside OneDrive
    safe_dir = os.path.join(os.path.expanduser("~"), ".penguin_squad")
    os.makedirs(safe_dir, exist_ok=True)
    env_path = os.path.join(safe_dir, ".env")
    lines: List[str] = []
    if os.path.exists(env_path):
        with open(env_path, encoding='utf-8') as f:
            lines = f.readlines()
    written = set()
    new_lines: List[str] = []
    for line in lines:
        k = line.split('=')[0].strip()
        if k in updates:
            new_lines.append(f"{k}={updates[k]}\n")
            written.add(k)
        else:
            new_lines.append(line)
    for k, v in updates.items():
        if k not in written:
            new_lines.append(f"{k}={v}\n")
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)
        for k, v in updates.items():
            if hasattr(Config, k):
                setattr(Config, k, v)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Menu builders  (text + Keyboard)
# ─────────────────────────────────────────────────────────────────────────────

def _main_menu_content() -> tuple:
    """Returns (text, Keyboard) for the main menu."""
    try:
        from core.bot_controller import get_controller
        ctrl     = get_controller()
        mode     = ctrl.get_mode()
        exchange = ctrl.get_active_exchange().upper()
        symbol   = ctrl.get_symbol()
        state    = "⏸ PAUSED" if ctrl.is_paused() else ("🛑 STOPPED" if ctrl.is_stopped() else "🟢 RUNNING")
    except Exception:
        mode, exchange, symbol, state = Config.MODE, getattr(Config, 'ACTIVE_EXCHANGE', 'MT5').upper(), Config.SYMBOLS[0] if Config.SYMBOLS else '?', "🟢 RUNNING"

    icon = _MODE_ICONS.get(mode, '⚙️')
    text = (
        f"🐧 <b>Madagascar Penguins</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"State:    <code>{state}</code>\n"
        f"Mode:     <code>{icon} {mode.upper()}</code>\n"
        f"Exchange: <code>{exchange}</code>\n"
        f"Symbol:   <code>{symbol}</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Choose an action:</i>"
    )
    kb = Keyboard(
        [("🟡 Demo",    "mode:demo"),    ("🔴 Live",     "mode:live"),    ("🔵 Backtest", "mode:backtest")],
        [("📡 Signals", "mode:signals"), ("🔬 Analysis", "mode:analysis")],
        [("⏸ Pause",    "bot:pause"),    ("▶️ Resume",   "bot:resume"),   ("🛑 Stop",    "bot:stop")],
        [("📊 Status",  "show:status"),  ("💰 Balance",  "show:balance"), ("📈 Trades",  "show:trades")],
        [("🌐 Connect Exchange",  "menu:connect")],
        [("⚙️ Settings", "menu:settings")],
    )
    return text, kb


def _settings_menu_content() -> tuple:
    exchange = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').upper()
    symbol   = Config.SYMBOLS[0] if Config.SYMBOLS else '?'
    risk     = getattr(Config, 'RISK_PERCENT', 0.005)
    text = (
        f"⚙️ <b>SETTINGS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Exchange: <code>{exchange}</code>\n"
        f"Symbol:   <code>{symbol}</code>\n"
        f"Risk:     <code>{risk*100:.2f}%</code>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    kb = Keyboard(
        [("🌐 Switch Exchange",  "menu:connect")],
        [("📡 Change Symbol",    "settings:symbol"),  ("⚡ Change Risk",  "settings:risk")],
        [("🔑 API Keys",         "menu:apikeys")],
        [("◀️ Main Menu",        "menu:main")],
    )
    return text, kb


def _connect_menu_content() -> tuple:
    current = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').upper()
    text = (
        f"🌐 <b>CHOOSE EXCHANGE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Current: <code>{current}</code>\n\n"
        f"Select the exchange you want to trade on.\n"
        f"You'll be guided through credential setup."
    )
    kb = Keyboard(
        [("1️⃣ MT5",      "connect:mt5"),    ("2️⃣ Binance",  "connect:binance")],
        [("3️⃣ Bybit",    "connect:bybit"),  ("4️⃣ OKX",      "connect:okx")],
        [("5️⃣ Kraken",   "connect:kraken"), ("6️⃣ KuCoin",   "connect:kucoin")],
        [("7️⃣ Gate.io",  "connect:gateio"), ("8️⃣ Bitget",   "connect:bitget")],
        [("9️⃣ MEXC",     "connect:mexc")],
        [("◀️ Back",      "menu:settings")],
    )
    return text, kb


def _apikeys_menu_content() -> tuple:
    text = (
        f"🔑 <b>API KEYS</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Tap a penguin to update their API key.\n\n"
        f"🎖 Skipper  → OpenAI      {_key_ok('OPENAI_API_KEY')}\n"
        f"🧠 Kowalski → Anthropic   {_key_ok('ANTHROPIC_API_KEY')}\n"
        f"🃏 Rico     → DeepSeek    {_key_ok('DEEPSEEK_API_KEY')}\n"
        f"🐧 Private  → Telegram    ✅ (token in ~/.penguin_squad/.env)\n"
    )
    kb = Keyboard(
        [(f"🎖 Skipper   {_key_ok('OPENAI_API_KEY')}",    "setkey:skipper")],
        [(f"🧠 Kowalski  {_key_ok('ANTHROPIC_API_KEY')}", "setkey:kowalski")],
        [(f"🃏 Rico      {_key_ok('DEEPSEEK_API_KEY')}",  "setkey:rico")],
        [("◀️ Settings", "menu:settings")],
    )
    return text, kb


def _status_content() -> tuple:
    try:
        from core.bot_controller import get_controller
        text = get_controller().get_status_text()
    except Exception:
        mode = Config.MODE
        icon = _MODE_ICONS.get(mode, '⚙️')
        text = f"🤖 <b>STATUS</b>\nMode: <code>{icon} {mode.upper()}</code>"
    kb = Keyboard(
        [("🔄 Refresh", "show:status"), ("💰 Balance", "show:balance")],
        [("◀️ Main Menu", "menu:main")],
    )
    return text, kb


# ─────────────────────────────────────────────────────────────────────────────
# ConversationManager — multi-step credential collection
# ─────────────────────────────────────────────────────────────────────────────

class ConversationManager:
    def __init__(self):
        self._state: Dict[str, dict] = {}

    def is_active(self, chat_id: str) -> bool:
        return chat_id in self._state

    def cancel(self, chat_id: str) -> tuple:
        self._state.pop(chat_id, None)
        return "🚫 Cancelled.", None

    def start_exchange(self, chat_id: str, exchange: str) -> tuple:
        """Start credential collection for a specific exchange. Returns (text, keyboard|None)."""
        self._state[chat_id] = {
            'flow':      'exchange',
            'exchange':  exchange,
            'step':      'mt5_login' if exchange == 'mt5' else 'api_key',
            'collected': {},
        }
        if exchange == 'mt5':
            current = Config.LOGIN or 'not set'
            return (
                f"🖥 <b>MT5 Setup</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Current login: <code>{current}</code>\n\n"
                f"Enter your <b>MT5 Login</b> (account number):\n"
                f"<i>(send current value or leave blank to keep)</i>",
                None
            )
        else:
            ex = exchange.upper()
            return (
                f"🔑 <b>{ex} API Setup</b>  (1/2)\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Enter your <b>{ex} API Key</b>:\n\n"
                f"🔒 <i>Stored locally on your PC only (~/.penguin_squad/.env) — never uploaded.</i>",
                None
            )

    def start_setkey(self, chat_id: str, penguin: str) -> tuple:
        if penguin not in _PENGUIN_KEYS:
            return f"❌ Unknown: {penguin}", None
        attr, label, service = _PENGUIN_KEYS[penguin]
        self._state[chat_id] = {'flow': 'setkey', 'step': 'enter', 'penguin': penguin, 'attr': attr}
        return (
            f"🔑 <b>Update {label} key</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Service: <code>{service}</code>\n\n"
            f"Enter the new API key:\n"
            f"🔒 <i>Stored locally on your PC only (~/.penguin_squad/.env) — never uploaded.</i>\n\n"
            f"Send /cancel to abort.",
            None
        )

    def handle(self, text: str, chat_id: str) -> tuple:
        """Process text. Returns (reply_text, keyboard|None)."""
        if chat_id not in self._state:
            return None, None
        state = self._state[chat_id]
        flow  = state['flow']
        if flow == 'exchange':
            return self._exchange_step(text, chat_id, state)
        if flow == 'setkey':
            return self._setkey_step(text, chat_id, state)
        return None, None

    # ── Exchange credential steps ─────────────────────────────────────────────

    def _exchange_step(self, text: str, chat_id: str, state: dict) -> tuple:
        step     = state['step']
        exchange = state['exchange']
        coll     = state['collected']

        if step == 'mt5_login':
            coll['MT5_LOGIN'] = text.strip() or Config.LOGIN
            state['step']     = 'mt5_password'
            return (
                f"✅ Login: <code>{coll['MT5_LOGIN']}</code>\n\n"
                f"Enter your <b>MT5 Password</b>:",
                None
            )
        if step == 'mt5_password':
            coll['MT5_PASSWORD'] = text.strip() or Config.PASSWORD
            state['step']        = 'mt5_server'
            current_server       = Config.SERVER or 'Exness-MT5Trial15'
            return (
                f"✅ Password saved.\n\n"
                f"Enter your <b>MT5 Server</b>:\n"
                f"<i>(current: {current_server})</i>",
                None
            )
        if step == 'mt5_server':
            coll['MT5_SERVER']    = text.strip() or Config.SERVER or 'Exness-MT5Trial15'
            coll['ACTIVE_EXCHANGE'] = 'mt5'
            return self._finish_exchange(chat_id, state)

        if step == 'api_key':
            if len(text.strip()) < 8:
                return "⚠️ Key too short. Try again or /cancel.", None
            coll[f'{exchange.upper()}_API_KEY'] = text.strip()
            state['step'] = 'api_secret'
            return (
                f"✅ API Key saved.\n\n"
                f"Enter your <b>{exchange.upper()} API Secret</b>  (2/2):",
                None
            )
        if step == 'api_secret':
            if len(text.strip()) < 8:
                return "⚠️ Secret too short. Try again or /cancel.", None
            coll[f'{exchange.upper()}_API_SECRET'] = text.strip()
            if exchange in _NEEDS_PASSPHRASE:
                state['step'] = 'passphrase'
                return (f"✅ Secret saved.\n\n{exchange.upper()} also needs a <b>Passphrase</b>:", None)
            coll['ACTIVE_EXCHANGE'] = exchange
            return self._finish_exchange(chat_id, state)

        if step == 'passphrase':
            coll[f'{exchange.upper()}_PASSPHRASE'] = text.strip()
            coll['ACTIVE_EXCHANGE'] = exchange
            return self._finish_exchange(chat_id, state)

        return "⚠️ Unexpected state. Send /cancel.", None

    def _finish_exchange(self, chat_id: str, state: dict) -> tuple:
        coll     = state['collected']
        exchange = state['exchange']
        _update_env(coll)
        test     = self._test_connection(exchange, coll)
        self._state.pop(chat_id, None)

        def mask(v): return v[:6] + '...' if len(v) > 6 else '***'
        saved = '\n'.join(
            f"  <code>{k}</code> = <code>{'***' if any(x in k for x in ('KEY','SECRET','PASSWORD','PASS')) else v}</code>"
            for k, v in coll.items()
        )
        text = (
            f"✅ <b>{exchange.upper()} configured!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{saved}\n\n"
            f"{test}"
        )
        kb = Keyboard(
            [("▶️ Start Demo", "mode:demo"), ("🔴 Start Live", "mode:live")],
            [("◀️ Main Menu",  "menu:main")],
        )
        return text, kb

    def _test_connection(self, exchange: str, coll: dict) -> str:
        try:
            if exchange == 'mt5':
                import MetaTrader5 as mt5
                login  = int(coll.get('MT5_LOGIN', Config.LOGIN))
                pw     = coll.get('MT5_PASSWORD', Config.PASSWORD)
                server = coll.get('MT5_SERVER', Config.SERVER)
                if not mt5.initialize():
                    return "⚠️ MT5 initialize() failed"
                ok = mt5.login(login, pw, server)
                if ok:
                    info = mt5.account_info()
                    bal  = f"${info.balance:.2f}" if info else "?"
                    mt5.shutdown()
                    return f"🟢 <b>MT5 connected!</b> Balance: <code>{bal}</code>"
                mt5.shutdown()
                return "🔴 MT5 login failed — check credentials."
            else:
                from core.exchange import get_exchange
                key    = coll.get(f'{exchange.upper()}_API_KEY', '')
                secret = coll.get(f'{exchange.upper()}_API_SECRET', '')
                passph = coll.get(f'{exchange.upper()}_PASSPHRASE', '')
                ex = get_exchange(exchange, api_key=key, api_secret=secret, passphrase=passph)
                if ex.connect():
                    bal = ex.get_balance()
                    ex.disconnect()
                    return f"🟢 <b>{exchange.upper()} connected!</b> Balance: <code>${bal:.2f}</code>"
                return f"🔴 Connection failed — check API keys."
        except Exception as e:
            return f"⚠️ Test error: <code>{str(e)[:100]}</code>"

    # ── Setkey step ───────────────────────────────────────────────────────────

    def _setkey_step(self, text: str, chat_id: str, state: dict) -> tuple:
        key     = text.strip()
        if len(key) < 8:
            return "⚠️ Key too short. Try again or /cancel.", None
        penguin = state['penguin']
        attr    = state['attr']
        _update_env({attr: key})
        try:
            setattr(Config, attr, key)
        except Exception:
            pass
        self._state.pop(chat_id, None)
        _, label, service = _PENGUIN_KEYS[penguin]
        text_out = (
            f"✅ <b>{label} key updated!</b>\n"
            f"<code>{attr}</code> = <code>{key[:8]}...</code>\n\n"
            f"<i>Active immediately.</i>"
        )
        kb = Keyboard(
            [("🔑 More Keys", "menu:apikeys"), ("◀️ Main Menu", "menu:main")],
        )
        return text_out, kb


# ─────────────────────────────────────────────────────────────────────────────
# Private
# ─────────────────────────────────────────────────────────────────────────────

class Private:
    name = "Private"

    def __init__(self):
        self._notifier    = get_notifier()
        self._cmd_bot     = TelegramCommandBot()
        self._conv        = ConversationManager()
        self._bot_started = False
        self._engine_ref  = None
        self._pending_live: Dict[str, bool] = {}
        self._register_all()

    # ── Engine ────────────────────────────────────────────────────────────────

    def set_engine(self, engine) -> None:
        self._engine_ref = engine
        try:
            from core.bot_controller import get_controller
            get_controller().set_engine(engine)
        except Exception:
            pass

    # ── LangGraph node ────────────────────────────────────────────────────────

    def run(self, state: Dict) -> Dict:
        approved = state.get("approved", False)
        action   = state.get("final_action", "skip")
        quality  = state.get("quality_score", 0.0)
        reasons  = state.get("rejection_reasons", [])
        exec_res = state.get("execution_result") or {}
        try:
            from core.bot_controller import get_controller
            if get_controller().get_mode() == 'analysis':
                return {"rico_notification": {"sent": False}}
        except Exception:
            pass
        logger.info(f"[Private] 💬 action={action} approved={approved} q={quality:.1f}")
        try:
            if approved and action in ("buy", "sell"):
                sig = state.get("signal") or {}
                self._notifier.trade_opened(
                    order_type=action,
                    symbol=state.get("symbol", Config.SYMBOLS[0]),
                    price=state.get("entry_price", sig.get("entry_price", 0.0)),
                    sl=state.get("sl_price",    sig.get("sl_price",    0.0)),
                    tp=state.get("tp_price",    sig.get("tp_price",    0.0)),
                    lot=exec_res.get("lot", 0.01),
                    ticket=exec_res.get("ticket", 0),
                    session=state.get("session", ""),
                    quality=quality,
                    strategy=(state.get("signal") or {}).get("strategy", ""),
                )
            else:
                if Config.MODE == 'signals' and (state.get("signal") or {}).get("type") in ("buy", "sell"):
                    sig = state.get("signal") or {}
                    self._notifier.send(
                        f"📡 <b>SIGNAL</b> (no execution)\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Direction : <code>{sig.get('type','?').upper()}</code>\n"
                        f"Symbol    : <code>{state.get('symbol','?')}</code>\n"
                        f"Entry     : <code>{state.get('entry_price',0):.2f}</code>\n"
                        f"SL        : <code>{state.get('sl_price',0):.2f}</code>\n"
                        f"TP        : <code>{state.get('tp_price',0):.2f}</code>\n"
                        f"Quality   : <code>{quality:.1f}</code>"
                    )
                else:
                    self._notifier.signal_skipped(reasons, quality)
        except Exception as exc:
            logger.error(f"[Private] send failed: {exc}")
        return {"rico_notification": {"sent": True, "action": action, "approved": approved,
                                      "ts": datetime.utcnow().isoformat()}}

    # ── Standalone notifiers ──────────────────────────────────────────────────

    def notify_trade_closed(self, *a): self._notifier.trade_closed(*a)
    def notify_startup(self, *a):      self._notifier.startup(*a)
    def notify_error(self, e):         self._notifier.error_alert(e)
    def notify_daily_summary(self, *a):self._notifier.daily_summary(*a)

    # ── Bot lifecycle ─────────────────────────────────────────────────────────

    def start_bot(self) -> None:
        if self._bot_started:
            return
        self._cmd_bot.register_text_handler(self._on_text)
        self._cmd_bot.start()
        self._bot_started = True
        logger.info("[Private] 🤖 Telegram bot started")

    def stop_bot(self) -> None:
        self._cmd_bot.stop()

    def register_command(self, cmd: str, handler: Callable) -> None:
        self._cmd_bot.register(cmd, handler)

    # ── Text fallback (conversations) ─────────────────────────────────────────

    def _on_text(self, text: str, chat_id: str) -> Optional[str]:
        if self._conv.is_active(chat_id):
            reply_text, kb = self._conv.handle(text, chat_id)
            if reply_text:
                if kb:
                    self._cmd_bot.send_menu(chat_id, reply_text, kb)
                    return None
                return reply_text
            return None
        return "💬 Send /menu to open the main menu."

    # ── Registration ──────────────────────────────────────────────────────────

    def _register_all(self) -> None:
        # Text commands
        cmds = {
            "/start":     self._cmd_menu,
            "/menu":      self._cmd_menu,
            "/help":      self._cmd_help,
            "/status":    self._cmd_status,
            "/balance":   self._cmd_balance,
            "/trades":    self._cmd_trades,
            "/mode":      self._cmd_mode,
            "/penguins":  self._cmd_penguins,
            "/connect":   self._cmd_connect,
            "/setkey":    self._cmd_setkey,
            "/cancel":    self._cmd_cancel,
            "/demo":      self._cmd_demo,
            "/live":      self._cmd_live,
            "/backtest":  self._cmd_backtest,
            "/signals":   self._cmd_signals,
            "/analysis":  self._cmd_analysis,
            "/pause":     self._cmd_pause,
            "/resume":    self._cmd_resume,
            "/stop":      self._cmd_stop,
            "/exchange":  self._cmd_exchange,
            "/exchanges": self._cmd_exchanges,
            "/symbol":    self._cmd_symbol,
            "/risk":      self._cmd_risk,
        }
        for cmd, fn in cmds.items():
            self._cmd_bot.register(cmd, fn)

        # Button callbacks
        self._cmd_bot.register_callback("menu:",     self._cb_menu)
        self._cmd_bot.register_callback("mode:",     self._cb_mode)
        self._cmd_bot.register_callback("bot:",      self._cb_bot)
        self._cmd_bot.register_callback("show:",     self._cb_show)
        self._cmd_bot.register_callback("connect:",  self._cb_connect)
        self._cmd_bot.register_callback("setkey:",   self._cb_setkey)
        self._cmd_bot.register_callback("settings:", self._cb_settings_action)
        self._cmd_bot.register_callback("cancel:",   self._cb_cancel)

    # ─────────────────────────────────────────────────────────────────────────
    # MENU CALLBACKS (button presses)
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_menu(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        target = data.split(":", 1)[1]  # e.g. "main", "settings", "connect", "apikeys"
        if target == "main":
            text, kb = _main_menu_content()
            self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)
        elif target == "settings":
            text, kb = _settings_menu_content()
            self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)
        elif target == "connect":
            text, kb = _connect_menu_content()
            self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)
        elif target == "apikeys":
            text, kb = _apikeys_menu_content()
            self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)

    def _cb_mode(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        mode = data.split(":", 1)[1]   # demo / live / signals / analysis
        if mode == "live":
            if not self._pending_live.get(chat_id):
                self._pending_live[chat_id] = True
                self._cmd_bot.send_menu(chat_id,
                    "⚠️ <b>LIVE TRADING</b>\nReal money will be used.\n\nPress <b>Confirm</b> to proceed.",
                    Keyboard(
                        [("🔴 CONFIRM LIVE", "mode:live_confirm")],
                        [("◀️ Cancel",       "menu:main")],
                    )
                )
                return
            return   # ignore stale click
        if mode == "live_confirm":
            self._pending_live.pop(chat_id, None)
            mode = "live"
        if mode == "backtest":
            if not self._pending_live.get(f"{chat_id}_bt"):
                self._pending_live[f"{chat_id}_bt"] = True
                self._cmd_bot.send_menu(chat_id,
                    "🔵 <b>BACKTEST</b>\nRuns historical simulation (no real orders).\n\nPress <b>Confirm</b> to start.",
                    Keyboard(
                        [("🔵 CONFIRM BACKTEST", "mode:backtest_confirm")],
                        [("◀️ Cancel",           "menu:main")],
                    )
                )
                return
            return
        if mode == "backtest_confirm":
            self._pending_live.pop(f"{chat_id}_bt", None)
            mode = "backtest"

        try:
            from core.bot_controller import get_controller
            ctrl = get_controller()
            ctrl.set_mode(mode)
            ctrl.resume()
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}")
            return
        self._notify_mode(chat_id, mode)
        text, kb = _main_menu_content()
        self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)

    def _cb_bot(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        action = data.split(":", 1)[1]
        try:
            from core.bot_controller import get_controller
            ctrl   = get_controller()
            result = {'pause': ctrl.pause, 'resume': ctrl.resume, 'stop': ctrl.stop}[action]()
            self._cmd_bot.answer_callback(cq_id, result[:50])
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}")
            return
        text, kb = _main_menu_content()
        self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)

    def _cb_show(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        target = data.split(":", 1)[1]
        if target == "status":
            text, kb = _status_content()
            self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)
        elif target == "balance":
            # Get fresh balance
            bal_text = self._cmd_balance([], chat_id) or "💰 N/A"
            kb = Keyboard(
                [("🔄 Refresh", "show:balance"), ("◀️ Menu", "menu:main")],
            )
            self._cmd_bot.edit_menu(chat_id, msg_id, bal_text, kb)
        elif target == "trades":
            tr_text = self._cmd_trades([], chat_id) or "📊 N/A"
            kb = Keyboard([("◀️ Menu", "menu:main")])
            self._cmd_bot.edit_menu(chat_id, msg_id, tr_text, kb)

    def _cb_connect(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        exchange = data.split(":", 1)[1]
        text, kb = self._conv.start_exchange(chat_id, exchange)
        # Replace the exchange picker with setup prompt (no buttons during wizard)
        if kb:
            self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)
        else:
            self._cmd_bot.edit_menu(
                chat_id, msg_id, text,
                Keyboard([("🚫 Cancel", "cancel:exchange")])
            )

    def _cb_setkey(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        penguin = data.split(":", 1)[1]
        text, _kb = self._conv.start_setkey(chat_id, penguin)
        self._cmd_bot.edit_menu(
            chat_id, msg_id, text,
            Keyboard([("🚫 Cancel", "cancel:setkey")])
        )

    def _cb_cancel(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        """Cancel any active conversation and return to main menu."""
        self._conv.cancel(chat_id)
        text, kb = _main_menu_content()
        self._cmd_bot.edit_menu(chat_id, msg_id, text, kb)

    def _cb_settings_action(self, data: str, chat_id: str, msg_id: int, cq_id: str) -> None:
        action = data.split(":", 1)[1]
        if action == "symbol":
            sym = Config.SYMBOLS[0] if Config.SYMBOLS else "?"
            self._cmd_bot.edit_menu(chat_id, msg_id,
                f"📡 <b>CHANGE SYMBOL</b>\nCurrent: <code>{sym}</code>\n\nType the new symbol and send it.\n"
                f"Examples: <code>BTC/USDT:USDT</code>  <code>USTECm</code>",
                Keyboard([("🚫 Cancel", "menu:settings")])
            )
        elif action == "risk":
            risk = getattr(Config, 'RISK_PERCENT', 0.005)
            self._cmd_bot.edit_menu(chat_id, msg_id,
                f"⚡ <b>CHANGE RISK %</b>\nCurrent: <code>{risk*100:.2f}%</code>\n\n"
                f"Type the new risk % and send it.\nExample: <code>0.5</code> (= 0.5%)",
                Keyboard([("🚫 Cancel", "menu:settings")])
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TEXT COMMANDS (fallback for users who prefer typing)
    # ─────────────────────────────────────────────────────────────────────────

    def _send_main_menu(self, chat_id: str) -> None:
        text, kb = _main_menu_content()
        self._cmd_bot.send_menu(chat_id, text, kb)

    def _cmd_menu(self, args, chat_id) -> None:
        self._send_main_menu(chat_id)

    def _cmd_help(self, args, chat_id) -> str:
        return (
            "🐧 <b>Madagascar Penguins — Commands</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Send /menu to open the visual menu 👆\n\n"
            "<b>Quick commands:</b>\n"
            "  /demo  /live  /signals  /analysis\n"
            "  /pause  /resume  /stop\n"
            "  /connect   — exchange setup\n"
            "  /setkey    — update LLM API key\n"
            "  /status  /balance  /trades\n"
            "  /symbol &lt;sym&gt;  /risk &lt;pct&gt;\n"
            "  /exchanges — list all exchanges"
        )

    def _cmd_status(self, args, chat_id) -> None:
        text, kb = _status_content()
        # Add live engine info
        eng = self._engine_ref
        if eng:
            try:
                pos = getattr(eng, "_current_position", None)
                text += (
                    f"\n\n📦 Open: <code>{pos.get('type','?').upper()} @ {pos.get('price',0):.2f}</code>"
                ) if pos else "\n📭 No open position"
            except Exception:
                pass
        self._cmd_bot.send_menu(chat_id, text, kb)

    def _cmd_balance(self, args, chat_id) -> str:
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            if info:
                ex = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').upper()
                return (
                    f"💰 <b>ACCOUNT ({ex})</b>\n"
                    f"Balance:  <code>${info.balance:.2f}</code>\n"
                    f"Equity:   <code>${info.equity:.2f}</code>\n"
                    f"Free:     <code>${info.margin_free:.2f}</code>"
                )
        except Exception:
            pass
        try:
            from core.exchange import get_exchange_from_config
            ex = get_exchange_from_config()
            if ex.connect():
                inf = ex.get_account_info()
                ex.disconnect()
                return (
                    f"💰 <b>ACCOUNT ({inf.get('exchange','?').upper()})</b>\n"
                    f"Balance: <code>${inf.get('balance',0):.2f}</code>\n"
                    f"Equity:  <code>${inf.get('equity',0):.2f}</code>"
                )
        except Exception:
            pass
        return f"💰 Balance: <code>${Config.INITIAL_BALANCE:.2f}</code>"

    def _cmd_trades(self, args, chat_id) -> str:
        try:
            st = getattr(self._engine_ref, "stats_tracker", None)
            if st and hasattr(st, "get_summary"):
                s  = st.get_summary()
                t, w, p = s.get("total_trades",0), s.get("wins",0), s.get("total_pnl",0.0)
                return (
                    f"📊 <b>TRADE STATS</b>\n"
                    f"Total: <code>{t}</code>\n"
                    f"Wins:  <code>{w} ({w/max(t,1)*100:.0f}%)</code>\n"
                    f"P&L:   <code>${p:+.2f}</code>"
                )
        except Exception:
            pass
        return "📊 Stats not available yet."

    def _cmd_mode(self, args, chat_id) -> str:
        m = Config.MODE
        return f"⚙️ Mode: <b>{_MODE_ICONS.get(m,'⚙️')} {m.upper()}</b>"

    def _cmd_penguins(self, args, chat_id) -> None:
        text, kb = _apikeys_menu_content()
        self._cmd_bot.send_menu(chat_id, text, kb)

    def _cmd_connect(self, args, chat_id) -> None:
        text, kb = _connect_menu_content()
        self._cmd_bot.send_menu(chat_id, text, kb)

    def _cmd_setkey(self, args, chat_id) -> None:
        if not args:
            text, kb = _apikeys_menu_content()
            self._cmd_bot.send_menu(chat_id, text, kb)
            return
        reply_text, kb = self._conv.start_setkey(chat_id, args[0])
        if kb:
            self._cmd_bot.send_menu(chat_id, reply_text, kb)
        else:
            self._cmd_bot.send_to(chat_id, reply_text)

    def _cmd_cancel(self, args, chat_id) -> None:
        text, _ = self._conv.cancel(chat_id)
        self._send_main_menu(chat_id)

    def _ctrl(self):
        from core.bot_controller import get_controller
        return get_controller()

    def _notify_mode(self, chat_id: str, mode: str) -> None:
        """Send a brief confirmation when mode is activated."""
        _icons = {'demo':'🟡','live':'🔴','signals':'📡','analysis':'🔬','backtest':'🔵'}
        _descs = {
            'demo':     'Paper trading — no real money.',
            'live':     '⚠️ Real money trading is active.',
            'signals':  'Signal alerts only — no execution.',
            'analysis': 'Market analysis mode — no trades.',
            'backtest': 'Historical simulation running...',
        }
        icon = _icons.get(mode, '⚙️')
        desc = _descs.get(mode, '')
        self._cmd_bot.send_to(chat_id,
            f"{icon} <b>{mode.upper()} MODE — Active</b>\n"
            f"▶️ {desc}\n"
            f"Use /pause to pause or /menu for options."
        )

    def _cmd_demo(self, args, chat_id) -> None:
        try:
            c = self._ctrl(); c.set_mode('demo'); c.resume()
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._notify_mode(chat_id, 'demo')
        self._send_main_menu(chat_id)

    def _cmd_live(self, args, chat_id) -> None:
        if not self._pending_live.get(chat_id):
            self._pending_live[chat_id] = True
            self._cmd_bot.send_menu(chat_id,
                "⚠️ <b>LIVE TRADING</b>\nReal money will be used.\n\nPress Confirm or send /live again.",
                Keyboard(
                    [("🔴 CONFIRM LIVE", "mode:live_confirm")],
                    [("◀️ Cancel",       "menu:main")],
                )
            )
            return
        self._pending_live.pop(chat_id, None)
        try:
            c = self._ctrl(); c.set_mode('live'); c.resume()
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._notify_mode(chat_id, 'live')
        self._send_main_menu(chat_id)

    def _cmd_signals(self, args, chat_id) -> None:
        try:
            c = self._ctrl(); c.set_mode('signals'); c.resume()
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._notify_mode(chat_id, 'signals')
        self._send_main_menu(chat_id)

    def _cmd_analysis(self, args, chat_id) -> None:
        try:
            c = self._ctrl(); c.set_mode('analysis'); c.resume()
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._notify_mode(chat_id, 'analysis')
        self._send_main_menu(chat_id)

    def _cmd_backtest(self, args, chat_id) -> None:
        if not self._pending_live.get(f"{chat_id}_bt"):
            self._pending_live[f"{chat_id}_bt"] = True
            self._cmd_bot.send_menu(chat_id,
                "🔵 <b>BACKTEST</b>\nRuns historical simulation (2024–2025).\nNo real orders placed.\n\nSend /backtest again to confirm.",
                Keyboard(
                    [("🔵 CONFIRM BACKTEST", "mode:backtest_confirm")],
                    [("◀️ Cancel",           "menu:main")],
                )
            )
            return
        self._pending_live.pop(f"{chat_id}_bt", None)
        try:
            c = self._ctrl(); c.set_mode('backtest'); c.resume()
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._notify_mode(chat_id, 'backtest')
        self._send_main_menu(chat_id)

    def _cmd_pause(self, args, chat_id) -> None:
        try:
            r = self._ctrl().pause()
            self._cmd_bot.send_to(chat_id, r)
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._send_main_menu(chat_id)

    def _cmd_resume(self, args, chat_id) -> None:
        try:
            r = self._ctrl().resume()
            self._cmd_bot.send_to(chat_id, r)
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}"); return
        self._send_main_menu(chat_id)

    def _cmd_stop(self, args, chat_id) -> str:
        try:
            return self._ctrl().stop()
        except Exception as e:
            return f"⚠️ {e}"

    def _cmd_exchange(self, args, chat_id) -> None:
        if not args:
            self._cmd_connect(args, chat_id)
            return
        try:
            r = self._ctrl().set_active_exchange(args[0])
            _update_env({'ACTIVE_EXCHANGE': args[0].lower()})
            self._cmd_bot.send_to(chat_id, r)
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}")
        self._send_main_menu(chat_id)

    def _cmd_exchanges(self, args, chat_id) -> None:
        try:
            from core.exchange import list_exchanges
            names = list_exchanges()
            lines = '\n'.join(f"  • <code>{n}</code>" for n in names)
            self._cmd_bot.send_menu(chat_id,
                f"🌐 <b>SUPPORTED EXCHANGES</b>\n{lines}\n\nFull setup: use /connect",
                Keyboard([("🌐 Connect Now", "menu:connect"), ("◀️ Back", "menu:main")])
            )
        except Exception as e:
            self._cmd_bot.send_to(chat_id, f"⚠️ {e}")

    def _cmd_symbol(self, args, chat_id) -> str:
        if not args:
            sym = Config.SYMBOLS[0] if Config.SYMBOLS else "?"
            return f"📡 Current symbol: <code>{sym}</code>\nUsage: <code>/symbol BTC/USDT:USDT</code>"
        try:
            r = self._ctrl().set_symbol(args[0])
            _update_env({'SYMBOLS': args[0]})
            return r
        except Exception as e:
            return f"⚠️ {e}"

    def _cmd_risk(self, args, chat_id) -> str:
        if not args:
            r = getattr(Config, 'RISK_PERCENT', 0.005)
            return f"⚡ Current risk: <code>{r*100:.2f}%</code>\nUsage: <code>/risk 0.5</code>"
        try:
            pct = float(args[0].replace('%', ''))
            if pct > 1: pct /= 100
            r = self._ctrl().set_risk(pct)
            _update_env({'RISK_PERCENT': str(pct)})
            return r
        except ValueError:
            return "❌ Usage: /risk 0.5  (percent)"
        except Exception as e:
            return f"⚠️ {e}"
