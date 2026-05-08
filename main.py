"""
main.py — Penguin Squad entry point.

Usage:
    python main.py                  # interactive setup if no keys, then run
    python main.py --mode demo
    python main.py --mode backtest
    python main.py --mode live
    python main.py --show-graph
    python main.py --no-orchestrator
"""

import argparse
import getpass
import logging
import os
import sys
import warnings
from logging.handlers import RotatingFileHandler

# silence LangGraph/LangChain deprecation noise
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langgraph")

# ── working directory ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── .env location — kept OUTSIDE OneDrive so API keys stay local ──────────
# Keys are stored in C:\Users\<user>\.penguin_squad\.env  (not synced)
# Fallback: project folder .env (legacy / first-run)
_HOME_ENV_DIR = os.path.join(os.path.expanduser("~"), ".penguin_squad")
_HOME_ENV     = os.path.join(_HOME_ENV_DIR, ".env")
_LOCAL_ENV    = os.path.join(SCRIPT_DIR, ".env")

def _get_env_path() -> str:
    """Return the canonical .env path (home dir preferred — outside OneDrive)."""
    os.makedirs(_HOME_ENV_DIR, exist_ok=True)
    # Migrate legacy .env from project folder to home dir (one-time)
    if not os.path.exists(_HOME_ENV) and os.path.exists(_LOCAL_ENV):
        import shutil
        shutil.copy2(_LOCAL_ENV, _HOME_ENV)
        # Replace local .env with a pointer comment so users know where it went
        with open(_LOCAL_ENV, "w", encoding="utf-8") as _f:
            _f.write(
                "# Keys have been moved to a safer location:\n"
                f"# {_HOME_ENV}\n"
                "# This file is intentionally left empty.\n"
            )
    return _HOME_ENV

# ── logging ───────────────────────────────────────────────────
LOG_PATH = os.path.join(SCRIPT_DIR, 'trading_bot.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_PATH, maxBytes=100 * 1024 * 1024,
                            backupCount=5, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
# silence the stream handler for the interactive setup phase
_stream_handler = next(
    (h for h in logging.root.handlers
     if isinstance(h, logging.StreamHandler)
     and not isinstance(h, RotatingFileHandler)), None
)


# ══════════════════════════════════════════════════════════════
# ASCII helpers (pure print — no imports needed yet)
# ══════════════════════════════════════════════════════════════

_COLORS = {
    "cyan":    "\033[96m", "magenta": "\033[95m",
    "green":   "\033[92m", "yellow":  "\033[93m",
    "white":   "\033[97m", "gray":    "\033[90m",
    "red":     "\033[91m", "reset":   "\033[0m",
}

def _c(text, color="white"):
    return _COLORS.get(color, "") + text + _COLORS["reset"]

def _penguins(sk="cyan", kw="magenta", rc="green", pr="yellow"):
    print(_c("       .---.     .---.     .---.     .---. ", sk))
    print(_c("      ( o o )   ", sk) +
          _c("( - - )   ", kw) +
          _c("( o o )   ", rc) +
          _c("( ^ ^ )", pr))
    print(_c("       \\ = /     ", sk) +
          _c("\\ w /     ", kw) +
          _c("\\ ~ /     ", rc) +
          _c("\\ _ / ", pr))
    print(_c("      SKIPPER   ", sk) +
          _c("KOWALSKI    ", kw) +
          _c("RICO    ", rc) +
          _c("PRIVATE", pr))

def _one(who):
    styles = {
        "skipper":  ("cyan",    "( o o )", "\\ = /", "SKIPPER"),
        "kowalski": ("magenta", "( - - )", "\\ w /", "KOWALSKI"),
        "rico":     ("green",   "( o o )", "\\ ~ /", "RICO"),
        "private":  ("yellow",  "( ^ ^ )", "\\ _ /", "PRIVATE"),
    }
    col, eyes, body, name = styles[who]
    print()
    print(_c(f"                  .---.", col))
    print(_c(f"                 {eyes}", col))
    print(_c(f"                  {body}", col))
    print(_c(f"                 {name}", col))
    print()

def _line():
    print(_c("  " + "=" * 46, "gray"))

def _ask(prompt, default=""):
    hint = f" [Enter = {default}]" if default else ""
    val = input(f"  {prompt}{hint} : ").strip()
    return val if val else default

def _ask_key(label):
    val = input(_c(f"  {label} API key (Enter to skip) : ", "gray")).strip()
    return val


# ══════════════════════════════════════════════════════════════
# Interactive first-run setup
# ══════════════════════════════════════════════════════════════

def _no_keys():
    """True when no LLM API key is configured at all."""
    from dotenv import load_dotenv
    load_dotenv(_get_env_path(), override=False)   # load from safe home-dir location
    return not any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("DEEPSEEK_API_KEY"),
    ])

def interactive_setup():
    """
    Shown on first run (or when no API keys are set).
    Lets the user configure agents without leaving the terminal.
    Updates .env in-place.
    """
    # silence logging during the interactive UI
    if _stream_handler:
        _stream_handler.setLevel(logging.CRITICAL)

    os.system("cls" if os.name == "nt" else "clear")
    print()
    _penguins()
    print()
    print(_c("        MADAGASCAR PENGUINS", "white"))
    _line()
    print()

    use = _ask("Set up AI agents? (y/n)", "n")
    if use.lower() != "y":
        if _stream_handler:
            _stream_handler.setLevel(logging.INFO)
        return   # rule-based — no changes needed

    # ---- SKIPPER -------------------------------------------
    os.system("cls" if os.name == "nt" else "clear")
    _one("skipper")
    _line()
    print(_c("  Choose provider for SKIPPER:", "white"))
    print(_c("    [1]  ChatGPT  (OpenAI GPT-4o)  -- platform.openai.com", "gray"))
    print(_c("    [2]  Gemini   (Google)          -- aistudio.google.com/apikey", "gray"))
    print(_c("    [0]  Skip     (rule-based)", "gray"))
    print()
    sc = _ask("Choice", "0")

    openai_key = ""
    google_key = ""
    skipper_provider = "none"

    if sc == "1":
        skipper_provider = "openai"
        openai_key = _ask_key("OpenAI")
    elif sc == "2":
        skipper_provider = "gemini"
        google_key = _ask_key("Google")

    # ---- KOWALSKI ------------------------------------------
    os.system("cls" if os.name == "nt" else "clear")
    _one("kowalski")
    _line()
    print(_c("  KOWALSKI  --  Claude  --  console.anthropic.com", "magenta"))
    print()
    anthropic_key = ""
    if _ask("Set up Kowalski? (y/n)", "y").lower() == "y":
        anthropic_key = _ask_key("Anthropic")

    # ---- RICO ----------------------------------------------
    os.system("cls" if os.name == "nt" else "clear")
    _one("rico")
    _line()
    print(_c("  RICO  --  Gemini  --  aistudio.google.com/apikey", "green"))
    print()
    if google_key:
        print(_c("  (using same Google key as Skipper)", "gray"))
        print()
    else:
        if _ask("Set up Rico? (y/n)", "y").lower() == "y":
            google_key = _ask_key("Google")

    # ---- PRIVATE -------------------------------------------
    os.system("cls" if os.name == "nt" else "clear")
    _one("private")
    _line()
    print(_c("  PRIVATE  --  DeepSeek  --  platform.deepseek.com", "yellow"))
    print()
    deepseek_key = ""
    if _ask("Set up Private? (y/n)", "y").lower() == "y":
        deepseek_key = _ask_key("DeepSeek")

    # ---- write keys into .env (safe location, outside OneDrive) ----
    env_path = _get_env_path()
    _update_env(env_path, {
        "OPENAI_API_KEY":    openai_key,
        "ANTHROPIC_API_KEY": anthropic_key,
        "GOOGLE_API_KEY":    google_key,
        "DEEPSEEK_API_KEY":  deepseek_key,
        "SKIPPER_PROVIDER":  skipper_provider,
        "ORCHESTRATOR_ENABLED": "True",
    })

    # reload env so Config picks up the new keys
    from dotenv import load_dotenv
    load_dotenv(_get_env_path(), override=True)

    # ---- summary -------------------------------------------
    os.system("cls" if os.name == "nt" else "clear")
    print()
    _penguins(
        "cyan"    if (openai_key or (skipper_provider == "gemini" and google_key)) else "gray",
        "magenta" if anthropic_key else "gray",
        "green"   if google_key    else "gray",
        "yellow"  if deepseek_key  else "gray",
    )
    print()
    _line()
    print()

    if _stream_handler:
        _stream_handler.setLevel(logging.INFO)


def _update_env(path: str, updates: dict):
    """Write / overwrite specific keys in .env, preserving all others."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

    written = set()
    new_lines = []
    for line in lines:
        key = line.split("=")[0].strip()
        if key in updates:
            if updates[key]:                      # only write non-empty
                new_lines.append(f"{key}={updates[key]}\n")
            else:
                new_lines.append(f"{key}=\n")    # keep blank
            written.add(key)
        else:
            new_lines.append(line)

    for key, val in updates.items():
        if key not in written:
            new_lines.append(f"{key}={val}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Penguin Squad Trading Bot")
    parser.add_argument('--mode',     choices=['backtest','demo','live','signals','analysis'], default=None)
    parser.add_argument('--exchange', default=None,
                        help='Exchange to use: mt5|binance|bybit|okx|kraken|kucoin|gateio|bitget|mexc')
    parser.add_argument('--show-graph',       action='store_true')
    parser.add_argument('--no-orchestrator',  action='store_true')
    parser.add_argument('--symbol',           default=None)
    parser.add_argument('--setup',            action='store_true',
                        help='Force re-run interactive agent setup')
    parser.add_argument('--setup-telegram',   action='store_true',
                        help='Force re-run Telegram + exchange setup wizard')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# Startup banner (after engine is ready)
# ══════════════════════════════════════════════════════════════

def print_banner(engine) -> None:
    from config import Config
    sk = "cyan"    if Config.OPENAI_API_KEY    or (Config.SKIPPER_PROVIDER == "gemini" and Config.GOOGLE_API_KEY) else "gray"
    kw = "magenta" if Config.ANTHROPIC_API_KEY else "gray"
    rc = "green"   if Config.GOOGLE_API_KEY    else "gray"
    pr = "yellow"  if Config.DEEPSEEK_API_KEY  else "gray"

    print()
    _penguins(sk, kw, rc, pr)
    print()
    _line()
    print(f"   {engine.symbol}  |  ⏸ PAUSED — waiting for Telegram"
          f"  |  Trades: {len(engine.stats_tracker.trades)}"
          f"  |  ML: {len(engine.online_model.trade_history)}")
    _line()
    print()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Telegram setup wizard (shown when no Telegram creds) ──
    # Must run before importing Config so .env is ready
    try:
        from setup_wizard import run_if_needed as _tg_setup, run_wizard as _tg_wizard
        if getattr(args, 'setup_telegram', False):
            _tg_wizard()
        else:
            _tg_setup()
    except Exception as _e:
        logging.warning(f"[STARTUP] Setup wizard skipped: {_e}")

    # ── Interactive setup (first run or --setup flag) ─────────
    if args.setup or _no_keys():
        interactive_setup()

    # ── Late imports (after .env is finalised) ────────────────
    from config import Config
    from core.engine import PersistentTradingEngine
    from orchestrator.graph import build_graph, print_graph_structure

    # ── Show graph and exit ───────────────────────────────────
    if args.show_graph:
        print_graph_structure()
        sys.exit(0)

    # ── CLI overrides ─────────────────────────────────────────
    if args.mode:
        Config.MODE          = args.mode
        Config.BACKTEST_MODE = args.mode == 'backtest'
        Config.PAPER_TRADING = args.mode == 'demo'
        Config.LIVE_MODE     = args.mode == 'live'

    if args.no_orchestrator:
        Config.ORCHESTRATOR_ENABLED = False

    if args.symbol:
        Config.SYMBOLS = [args.symbol]

    if args.exchange:
        Config.ACTIVE_EXCHANGE = args.exchange.lower()

    # ── Graph flow ────────────────────────────────────────────
    if Config.ORCHESTRATOR_ENABLED:
        print_graph_structure()

    # ── Engine (no exchange connection yet — waits for Telegram) ─
    # Exchange connects AFTER user picks mode via Telegram
    symbol = Config.SYMBOLS[0] if Config.SYMBOLS else 'USTECm'
    logging.info("[STARTUP] Initialising trading engine (offline — exchange connects after mode is chosen)...")
    try:
        engine = PersistentTradingEngine(symbol=symbol)
    except Exception as exc:
        logging.critical(f"[STARTUP] Engine init failed: {exc}")
        sys.exit(1)

    # ── Graph ─────────────────────────────────────────────────
    if Config.ORCHESTRATOR_ENABLED:
        try:
            engine._graph = build_graph(engine=engine)
            logging.info("[STARTUP] Penguin Squad graph ready")
        except Exception as exc:
            logging.warning(f"[STARTUP] Graph pre-build failed: {exc}")

    # ── Banner ────────────────────────────────────────────────
    print_banner(engine)

    # ── Notify Telegram: show full menu immediately ───────────
    try:
        from utils.telegram_notifier import get_notifier
        from agents.private import _main_menu_content
        _n = get_notifier()
        # Use the same full menu content as /menu command
        _menu_text, _menu_kb = _main_menu_content()
        # Prepend a "Ready!" header
        _ex = getattr(Config, 'ACTIVE_EXCHANGE', 'MT5').upper()
        _ready_text = (
            "🐧 <b>Madagascar Penguins — Ready!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Exchange: <code>{_ex}</code>\n"
            f"Symbol:   <code>{symbol}</code>\n"
            "State:    <code>⏸ PAUSED</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<i>Choose a mode to start trading:</i>"
        )
        _n.send_menu(_n.chat_id, _ready_text, _menu_kb)
        logging.info("[STARTUP] Telegram notified — waiting for user command")
    except Exception as _te:
        logging.warning(f"[STARTUP] Telegram ready-notify failed: {_te}")

    # ── Run ───────────────────────────────────────────────────
    logging.info(f"[BOT_START] Mode={Config.get_file_prefix().upper()} | "
                 f"Orchestrator={'ON' if Config.ORCHESTRATOR_ENABLED else 'OFF'}")
    try:
        engine.run()
    except KeyboardInterrupt:
        logging.info("[SHUTDOWN] Stopped by user")
        engine.shutdown()
    except SystemExit as exc:
        engine.shutdown()
        sys.exit(int(str(exc)) if str(exc).isdigit() else 1)
    except Exception as exc:
        logging.critical(f"[FATAL] {exc}", exc_info=True)
        try:
            engine.shutdown()
        except Exception:
            pass
        sys.exit(1)


if __name__ == '__main__':
    main()
