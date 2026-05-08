"""
setup_wizard.py — First-run terminal setup for Penguin Squad.

Collects everything in one screen:
  1. Telegram Bot Token  (required)
  2. Telegram Chat ID    (required)
  3. Skipper  → OpenAI API key     (optional — Enter to skip)
  4. Rico     → DeepSeek API key   (optional — Enter to skip)
  5. Kowalski → Anthropic API key  (optional — Enter to skip)

All API keys can also be added / updated any time from Telegram:
  /setkey  →  🔑 API Keys menu

Run manually:  python setup_wizard.py
Auto-runs:     called by main.py when Telegram credentials are missing
"""

import os
import time


# ── ASCII art ──────────────────────────────────────────────────────────────────

_HEADER = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         🐧  P E N G U I N   S Q U A D  —  S E T U P  🐧    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

_PRIVATE = r"""
           .--.
          |^ ^ |      Hi! I'm Private 💬
          | __ |
         //    \\     I'll be your Telegram bot.
        (|      |)    Give me your token and I'll
       /'|______|'\   take care of the rest!
       \__________/
"""

_PENGUINS = r"""
   .---.     .---.     .---.     .---.
  ( o o )   ( - - )   ( ~ ~ )   ( ^ ^ )
   \ = /     \ w /     \ o /     \ _ /
  SKIPPER  KOWALSKI    RICO     PRIVATE
  GPT-4o    Claude   DeepSeek  Telegram
"""

_DONE = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ✅  Setup complete!  Bot is ready to launch.               ║
║                                                              ║
║   Open Telegram and send commands to your bot:               ║
║                                                              ║
║   /setkey     ← add or update penguin API keys at any time   ║
║                   🎖 Skipper  → OpenAI                       ║
║                   🃏 Rico     → DeepSeek                     ║
║                   🧠 Kowalski → Anthropic                    ║
║                                                              ║
║   /connect    ← set up exchange (MT5 / Binance / Bybit / …) ║
║   /demo       ← start paper trading                         ║
║   /live       ← start live trading                          ║
║   /signals    ← signal alerts only (no execution)           ║
║   /help       ← full command list                           ║
║                                                              ║
║   Everything — exchange, risk, symbols, API keys —           ║
║   is managed from Telegram. No terminal needed again!        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

# penguin, env_key, label, service, url
_PENGUIN_KEYS = [
    ('SKIPPER',  'OPENAI_API_KEY',    '🎖 Skipper',  'OpenAI',    'platform.openai.com/api-keys'),
    ('RICO',     'DEEPSEEK_API_KEY',  '🃏 Rico',     'DeepSeek',  'platform.deepseek.com'),
    ('KOWALSKI', 'ANTHROPIC_API_KEY', '🧠 Kowalski', 'Anthropic', 'console.anthropic.com/settings/keys'),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def _prompt(label: str) -> str:
    print(f"\n  {label}")
    return input("  ❯ ").strip()


def _find_env() -> str:
    """
    Store .env in user's home directory (~/.penguin_squad/.env)
    so it is never uploaded to OneDrive / iCloud / Dropbox.
    Falls back to project folder for legacy compatibility.
    """
    safe_dir = os.path.join(os.path.expanduser("~"), ".penguin_squad")
    os.makedirs(safe_dir, exist_ok=True)
    safe_env = os.path.join(safe_dir, ".env")

    # One-time migration: move .env from project folder to home dir
    project_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(safe_env) and os.path.exists(project_env):
        import shutil
        shutil.copy2(project_env, safe_env)
        with open(project_env, "w", encoding="utf-8") as _f:
            _f.write(
                "# Keys have been moved to a safer location:\n"
                f"# {safe_env}\n"
                "# This file is intentionally left empty.\n"
            )
    return safe_env


def _read_env() -> dict:
    result = {}
    path = _find_env()
    if not os.path.exists(path):
        return result
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            result[k.strip()] = v.strip().split('#')[0].strip()
    return result


def _write_env(updates: dict) -> None:
    path = _find_env()
    lines: list = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
    written = set()
    new_lines = []
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
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def _telegram_ok() -> bool:
    env   = _read_env()
    token = env.get('TELEGRAM_BOT_TOKEN', '')
    cid   = env.get('TELEGRAM_CHAT_ID', '')
    if not token or token.startswith('123456') or 'ABCD' in token:
        return False
    if not cid or cid.startswith('-1001234'):
        return False
    return True


def _mask(v: str) -> str:
    return v[:8] + '...' if len(v) > 8 else '***'


# ── Main wizard ────────────────────────────────────────────────────────────────

def run_wizard() -> None:
    _clear()
    print(_HEADER)
    time.sleep(0.2)
    print(_PRIVATE)
    time.sleep(0.3)
    print(_PENGUINS)
    time.sleep(0.2)

    print("""
  Welcome to Penguin Squad!

  This one-time setup collects:
    • Telegram bot credentials  (required — this is how you control the bot)
    • LLM API keys              (optional — you can also add them from Telegram)

  Press Enter to begin...
""")
    input("  ❯ ")

    env     = _read_env()
    updates = {}

    # ── TELEGRAM (required) ────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  📱  TELEGRAM  (required)")
    print("═" * 64)
    print("""
  HOW TO GET YOUR BOT TOKEN:
    1. Open Telegram → search @BotFather
    2. Send  /newbot  and follow the prompts
    3. You'll get a token like:  123456789:ABCDefGhIJKlmNoPQRsTUVwxyZ

  HOW TO GET YOUR CHAT ID:
    • Direct message: start a chat with your bot, then visit
        https://api.telegram.org/bot<TOKEN>/getUpdates
      and look for  "chat":{"id": ...}
    • Group: add @userinfobot to your group — it shows the group ID
      (Group IDs look like: -1001234567890)
""")

    current_token = env.get('TELEGRAM_BOT_TOKEN', '')
    if current_token:
        print(f"  (current token: {_mask(current_token)}  — press Enter to keep)")

    token = _prompt("Bot Token:")
    if not token and current_token:
        token = current_token
        print(f"  ↩  Keeping existing token.")
    while not token or ':' not in token:
        print("  ⚠️  Doesn't look right — a token contains a ':' character.")
        token = _prompt("Bot Token:")

    current_cid = env.get('TELEGRAM_CHAT_ID', '')
    if current_cid:
        print(f"\n  (current chat ID: {current_cid}  — press Enter to keep)")

    chat_id = _prompt("Chat ID:")
    if not chat_id and current_cid:
        chat_id = current_cid
        print(f"  ↩  Keeping existing chat ID.")
    while not chat_id:
        chat_id = _prompt("Chat ID can't be empty:")

    updates['TELEGRAM_BOT_TOKEN'] = token
    updates['TELEGRAM_CHAT_ID']   = chat_id
    updates['TELEGRAM_ENABLED']   = 'True'

    # ── EXCHANGE SETUP ─────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  🌐  EXCHANGE SETUP")
    print("═" * 64)
    print("""
  Which exchange will you use?

    [1]  MetaTrader 5   (Forex / CFDs — Exness, IC Markets, Pepperstone …)
    [2]  Bybit          (Crypto futures / spot)
    [3]  Binance        (Crypto futures / spot)
    [4]  OKX            (Crypto futures / spot)
    [5]  Kraken         (Crypto spot)
    [0]  Skip for now   (configure later from Telegram)
""")
    exc_choice = _prompt("Choice", "0")

    _exchange_map = {
        '1': 'mt5', '2': 'bybit', '3': 'binance',
        '4': 'okx', '5': 'kraken',
    }
    chosen_exchange = _exchange_map.get(exc_choice, '')

    if chosen_exchange:
        updates['ACTIVE_EXCHANGE'] = chosen_exchange

    if chosen_exchange == 'mt5':
        print("""
  MetaTrader 5 setup:
    • You need MetaTrader 5 desktop app installed from your broker.
    • Download: https://www.metatrader5.com/en/download
    • Or directly from your broker (Exness, IC Markets, etc.)
""")
        mt5_login  = _prompt("MT5 Login (account number):")
        mt5_pass   = _prompt("MT5 Password:")
        mt5_server = _prompt("MT5 Server (e.g. Exness-MT5Trial15):")
        if mt5_login:  updates['MT5_LOGIN']    = mt5_login
        if mt5_pass:   updates['MT5_PASSWORD'] = mt5_pass
        if mt5_server: updates['MT5_SERVER']   = mt5_server

    elif chosen_exchange in ('bybit', 'binance', 'okx', 'kraken'):
        name = chosen_exchange.upper()
        print(f"""
  {name} API setup:
    1. Log in to your {name} account
    2. Go to API Management and create a new key
    3. Enable: Read + Trade  (do NOT enable Withdraw)
    4. Save the API Key and Secret Key
""")
        api_key    = _prompt(f"{name} API Key:")
        api_secret = _prompt(f"{name} API Secret:")
        key_map = {
            'bybit':   ('BYBIT_API_KEY',   'BYBIT_API_SECRET'),
            'binance': ('BINANCE_API_KEY', 'BINANCE_API_SECRET'),
            'okx':     ('OKX_API_KEY',     'OKX_API_SECRET'),
            'kraken':  ('KRAKEN_API_KEY',  'KRAKEN_API_SECRET'),
        }
        k, s = key_map[chosen_exchange]
        if api_key:    updates[k] = api_key
        if api_secret: updates[s] = api_secret

        if chosen_exchange == 'okx':
            okx_pp = _prompt("OKX Passphrase:")
            if okx_pp: updates['OKX_PASSPHRASE'] = okx_pp

    # ── PENGUIN API KEYS (optional) ────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  🐧  PENGUIN API KEYS  (optional)")
    print("═" * 64)
    print("""
  Press Enter to skip any key you don't have yet.
  You can add or update them any time from Telegram:
    → Send  /setkey  to your bot
""")

    for penguin, env_key, label, service, url in _PENGUIN_KEYS:
        current = env.get(env_key, '')
        status  = f"  (current: {_mask(current)}  — Enter to keep)" if current else "  (Enter to skip)"
        print(f"  {label}  →  {service}")
        print(f"     {url}{status}")
        val = _prompt(f"{service} API Key:")
        if val:
            updates[env_key] = val
        # if empty + existing → keep existing (don't overwrite with empty)

    # ── Save ───────────────────────────────────────────────────────────────────
    updates['BOT_MODE'] = env.get('BOT_MODE', 'demo')
    _write_env(updates)

    print("\n" + "─" * 64)
    print("  💾  Saved to .env:")
    for k, v in updates.items():
        display = _mask(v) if any(x in k for x in ('KEY', 'SECRET', 'TOKEN', 'PASSWORD')) else v
        print(f"    {k} = {display}")
    print("─" * 64)

    print(_DONE)
    input("  Press Enter to launch the bot...")


# ── Auto-run ───────────────────────────────────────────────────────────────────

def run_if_needed() -> None:
    """Called at bot startup. Runs wizard only when Telegram credentials are missing."""
    if not _telegram_ok():
        run_wizard()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass


if __name__ == '__main__':
    run_wizard()
