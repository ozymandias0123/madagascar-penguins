"""
utils/telegram_notifier.py
Telegram Bot — Notifications + Commands + Inline Menus

Three layers:
  TelegramNotifier   — send messages, keyboards, edit messages
  TelegramCommandBot — long-poll: /commands, button callbacks, plain text
  Keyboard           — builder helper for inline keyboards

Keyboard example:
    kb = Keyboard(
        [("▶️ Demo", "mode:demo"), ("🔴 Live", "mode:live")],
        [("📊 Status", "show:status"), ("⚙️ Settings", "menu:settings")],
    )
    notifier.send_menu(chat_id, "Choose:", kb)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import requests

from config import Config

logger = logging.getLogger(__name__)

E = {
    "bot":     "🤖", "start":   "🚀", "win":     "🏆", "loss":    "💥",
    "buy":     "🟢", "sell":    "🔴", "skip":    "⏭",  "warn":    "⚠️",
    "price":   "💰", "sl":      "🛑", "tp":      "🎯", "lot":     "📦",
    "time":    "⏱",  "stat":    "📊", "news":    "📰", "penguin": "🐧",
    "check":   "✅", "cross":   "❌", "fire":    "🔥", "info":    "ℹ️",
    "signal":  "📡", "risk":    "🛡", "quality": "⭐", "key":     "🔑",
    "lock":    "🔒", "globe":   "🌐", "gear":    "⚙️", "back":    "◀️",
    "refresh": "🔄", "pause":   "⏸",  "resume":  "▶️", "stop":    "🛑",
}


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard builder
# ─────────────────────────────────────────────────────────────────────────────

class Keyboard:
    """
    Builds a Telegram InlineKeyboardMarkup dict.

    Usage:
        kb = Keyboard(
            [("▶️ Demo", "mode:demo"), ("🔴 Live", "mode:live")],
            [("◀️ Back", "menu:main")],
        )
        # kb.markup → {"inline_keyboard": [...]}
    """

    def __init__(self, *rows: List[Tuple[str, str]]):
        self._rows = rows

    @property
    def markup(self) -> dict:
        return {
            "inline_keyboard": [
                [{"text": text, "callback_data": data} for text, data in row]
                for row in self._rows
            ]
        }

    @staticmethod
    def single(text: str, data: str) -> "Keyboard":
        return Keyboard([(text, data)])


# ─────────────────────────────────────────────────────────────────────────────
# TelegramNotifier
# ─────────────────────────────────────────────────────────────────────────────

class TelegramNotifier:
    """Sends formatted HTML messages (with optional inline keyboards)."""

    API = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token   = token   or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.warning("[Telegram] Token/ChatID not set — notifications disabled")

    # ── Raw HTTP ──────────────────────────────────────────────────────────────

    def _post(self, method: str, payload: dict) -> Optional[dict]:
        if not self.token:
            return None
        url = self.API.format(token=self.token, method=method)
        try:
            r    = requests.post(url, json=payload, timeout=10)
            data = r.json()
            if not data.get("ok"):
                desc = data.get('description', '')
                # Suppress harmless "same content" error (user pressed button twice)
                if 'message is not modified' not in desc:
                    logger.warning(f"[Telegram] {method} error: {desc}")
            return data
        except Exception as exc:
            logger.error(f"[Telegram] {method} failed: {exc}")
            return None

    # ── Send helpers ──────────────────────────────────────────────────────────

    def send(self, text: str, parse_mode: str = "HTML",
             silent: bool = False) -> Optional[dict]:
        """Send plain text to the configured chat."""
        return self._post("sendMessage", {
            "chat_id":              self.chat_id,
            "text":                 text,
            "parse_mode":           parse_mode,
            "disable_notification": silent,
        })

    def send_to(self, chat_id: str, text: str,
                parse_mode: str = "HTML") -> Optional[dict]:
        """Send plain text to a specific chat_id."""
        return self._post("sendMessage", {
            "chat_id":    chat_id,
            "text":       text,
            "parse_mode": parse_mode,
        })

    def send_menu(self, chat_id: str, text: str,
                  keyboard: Keyboard,
                  parse_mode: str = "HTML") -> Optional[dict]:
        """Send a message with an inline keyboard."""
        return self._post("sendMessage", {
            "chat_id":      chat_id,
            "text":         text,
            "parse_mode":   parse_mode,
            "reply_markup": keyboard.markup,
        })

    def edit_menu(self, chat_id: str, message_id: int,
                  text: str, keyboard: Keyboard,
                  parse_mode: str = "HTML") -> Optional[dict]:
        """
        Edit an existing message and its keyboard in-place.
        Used so button presses update the same message instead of sending new ones.
        """
        return self._post("editMessageText", {
            "chat_id":      chat_id,
            "message_id":   message_id,
            "text":         text,
            "parse_mode":   parse_mode,
            "reply_markup": keyboard.markup,
        })

    def answer_callback(self, callback_query_id: str,
                        text: str = "", alert: bool = False) -> None:
        """Acknowledge a button press (removes the loading spinner)."""
        self._post("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text":              text,
            "show_alert":        alert,
        })

    # ── Pre-built notification templates ─────────────────────────────────────

    def startup(self, symbol: str, balance: float,
                prev_trades: int, mode: str) -> None:
        exchange = getattr(Config, 'ACTIVE_EXCHANGE', 'mt5').upper()
        text = (
            f"{E['start']} <b>PENGUIN SQUAD — BOT STARTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{E['stat']} Mode:      <code>{mode.upper()}</code>\n"
            f"{E['globe']} Exchange:  <code>{exchange}</code>\n"
            f"{E['penguin']} Symbol:    <code>{symbol}</code>\n"
            f"{E['price']} Balance:   <code>${balance:.2f}</code>\n"
            f"{E['info']} Trades:    <code>{prev_trades} historical</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Skipper, Kowalski, Rico, Private — all on deck.</i>"
        )
        kb = Keyboard(
            [("📊 Status", "show:status"), ("💰 Balance", "show:balance")],
            [("⚙️ Main Menu", "menu:main")],
        )
        if self.chat_id:
            self.send_menu(self.chat_id, text, kb)
        else:
            self.send(text)

    def trade_opened(self, order_type: str, symbol: str,
                     price: float, sl: float, tp: float,
                     lot: float, ticket: int,
                     session: str, quality: float,
                     strategy: str = "") -> None:
        e   = E["buy"] if order_type.lower() == "buy" else E["sell"]
        rr  = abs(tp - price) / max(abs(price - sl), 0.001)
        strat = f"\n{E['signal']} Strategy: <code>{strategy}</code>" if strategy else ""
        text = (
            f"{e} <b>TRADE OPENED — {order_type.upper()} {symbol}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{E['price']} Entry:   <code>{price:.2f}</code>\n"
            f"{E['sl']} SL:      <code>{sl:.2f}</code>\n"
            f"{E['tp']} TP:      <code>{tp:.2f}</code>\n"
            f"{E['stat']} RR:      <code>1 : {rr:.1f}</code>\n"
            f"{E['lot']} Lot:     <code>{lot:.3f}</code>\n"
            f"{E['time']} Session: <code>{session.title()}</code>\n"
            f"{E['quality']} Quality: <code>{quality:.1f}/10</code>"
            f"{strat}"
        )
        self.send(text)

    def trade_closed(self, order_type: str, symbol: str,
                     net_pnl: float, profit_r: float,
                     exit_price: float, duration_min: float,
                     trade_num: int) -> None:
        won  = net_pnl > 0
        e    = E["win"] if won else E["loss"]
        text = (
            f"{e} <b>TRADE {'WON' if won else 'LOST'} ({profit_r:+.1f}R)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{'✅' if won else '❌'} {order_type.upper()} {symbol}\n"
            f"{E['price']} P&L:      <code>${net_pnl:+.2f}</code>\n"
            f"{E['stat']} R-Multiple:<code>{profit_r:+.2f}R</code>\n"
            f"{E['tp']} Exit:     <code>{exit_price:.2f}</code>\n"
            f"{E['time']} Duration: <code>{duration_min:.1f} min</code>\n"
            f"#{trade_num}"
        )
        self.send(text)

    def signal_skipped(self, reasons: List[str], quality: float) -> None:
        lines = "\n".join(f"  • {r}" for r in reasons[:5]) or "  • No signal"
        self.send(
            f"{E['skip']} <b>SIGNAL SKIPPED</b>  (quality {quality:.1f})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n{lines}",
            silent=True,
        )

    def orchestrator_summary(self, state: dict) -> None:
        approved = state.get("approved", False)
        action   = state.get("final_action", "skip")
        quality  = state.get("quality_score", 0.0)
        ma       = state.get("market_analysis") or {}
        ra       = state.get("risk_assessment") or {}
        val      = state.get("validation") or {}
        e_action = E["buy"] if action=="buy" else (E["sell"] if action=="sell" else E["skip"])
        e_ok     = E["check"] if approved else E["cross"]
        self.send(
            f"{E['penguin']} <b>PENGUIN SQUAD DECISION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{e_action} Action:    <b>{action.upper()}</b>\n"
            f"{e_ok} Approved:  <code>{'YES' if approved else 'NO'}</code>\n"
            f"{E['quality']} Quality:   <code>{quality:.1f}/10</code>\n"
            f"{E['stat']} Market:    <code>{ma.get('recommendation','N/A')}</code>\n"
            f"{E['risk']} Risk:      <code>{ra.get('risk_level','N/A')}</code>\n"
            f"{E['check']} Validator: <code>{val.get('confidence',0)}%</code>",
            silent=(not approved),
        )

    def error_alert(self, error: str) -> None:
        self.send(f"{E['warn']} <b>BOT ERROR</b>\n<code>{error[:300]}</code>")

    def daily_summary(self, trades: int, wins: int,
                      total_pnl: float, balance: float) -> None:
        wr = wins / max(trades, 1) * 100
        e  = "📈" if total_pnl >= 0 else "📉"
        self.send(
            f"{e} <b>DAILY SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{E['stat']} Trades:  <code>{trades}</code>\n"
            f"{E['check']} Wins:    <code>{wins}/{trades} ({wr:.0f}%)</code>\n"
            f"{E['price']} P&L:     <code>${total_pnl:+.2f}</code>\n"
            f"💼 Balance: <code>${balance:.2f}</code>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TelegramCommandBot  (commands + callbacks + conversations)
# ─────────────────────────────────────────────────────────────────────────────

class TelegramCommandBot:
    """
    Long-polling bot with three dispatch layers:
      1. /commands       → command handlers
      2. button presses  → callback handlers  (callback_data → handler)
      3. plain text      → text handler       (for multi-step conversations)
    """

    POLL_TIMEOUT = 30
    POLL_DELAY   = 1

    def __init__(self, token: str = ""):
        self.token              = token or Config.TELEGRAM_BOT_TOKEN
        self._handlers:  Dict[str, Callable] = {}    # /cmd → fn(args, chat_id)
        self._callbacks: Dict[str, Callable] = {}    # callback_data → fn(data, chat_id, msg_id, cq_id)
        self._text_handler: Optional[Callable] = None
        self._offset   = 0
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._bot      = TelegramNotifier(token=self.token)

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, command: str, handler: Callable) -> None:
        """Register a /command handler.  fn(args: list[str], chat_id: str) -> str|None"""
        self._handlers[command.lower()] = handler

    def register_callback(self, data_prefix: str, handler: Callable) -> None:
        """
        Register a button callback handler.
        data_prefix  — exact callback_data OR prefix ending in ':'
        fn(data: str, chat_id: str, message_id: int, cq_id: str) -> str|None
        """
        self._callbacks[data_prefix] = handler

    def register_text_handler(self, handler: Callable) -> None:
        """Register fallback for plain-text messages.  fn(text, chat_id) -> str|None"""
        self._text_handler = handler

    # ── Sending helpers (exposed so agents can use them) ──────────────────────

    def send_to(self, chat_id: str, text: str) -> None:
        self._bot.send_to(chat_id, text)

    def send_menu(self, chat_id: str, text: str, keyboard: Keyboard) -> Optional[dict]:
        return self._bot.send_menu(chat_id, text, keyboard)

    def edit_menu(self, chat_id: str, message_id: int,
                  text: str, keyboard: Keyboard) -> Optional[dict]:
        return self._bot.edit_menu(chat_id, message_id, text, keyboard)

    def answer_callback(self, cq_id: str, text: str = "") -> None:
        self._bot.answer_callback(cq_id, text)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if not self.token:
            logger.warning("[TelegramBot] No token — disabled")
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="PrivateBotPoller"
        )
        self._thread.start()
        logger.info("[TelegramBot] Private bot started (polling)")

    def stop(self) -> None:
        self._running = False

    # ── Polling ───────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while self._running:
            try:
                updates = self._get_updates()
                for upd in updates:
                    self._handle_update(upd)
            except Exception as exc:
                logger.debug(f"[TelegramBot] poll error: {exc}")
            time.sleep(self.POLL_DELAY)

    def _get_updates(self) -> list:
        url = TelegramNotifier.API.format(token=self.token, method="getUpdates")
        r   = requests.get(url, params={
            "offset":         self._offset,
            "timeout":        self.POLL_TIMEOUT,
            "allowed_updates": ["message", "callback_query"],
        }, timeout=self.POLL_TIMEOUT + 5)
        data    = r.json()
        updates = data.get("result", [])
        if updates:
            self._offset = updates[-1]["update_id"] + 1
        return updates

    def _handle_update(self, upd: dict) -> None:
        # ── Button press (callback_query) ─────────────────────────────────────
        if "callback_query" in upd:
            cq      = upd["callback_query"]
            cq_id   = cq["id"]
            data    = cq.get("data", "")
            chat_id = str(cq.get("message", {}).get("chat", {}).get("id", ""))
            msg_id  = cq.get("message", {}).get("message_id", 0)

            # Acknowledge immediately (removes spinner)
            self._bot.answer_callback(cq_id)

            # Find handler: exact match first, then prefix match
            handler = self._callbacks.get(data)
            if handler is None:
                for prefix, h in self._callbacks.items():
                    if prefix.endswith(':') and data.startswith(prefix):
                        handler = h
                        break

            if handler:
                try:
                    reply = handler(data, chat_id, msg_id, cq_id)
                    if reply and isinstance(reply, str):
                        self._bot.send_to(chat_id, reply)
                except Exception as exc:
                    logger.error(f"[TelegramBot] callback {data}: {exc}")
            else:
                logger.debug(f"[TelegramBot] unhandled callback: {data}")
            return

        # ── Text / command message ─────────────────────────────────────────────
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            return
        text    = (msg.get("text") or "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not text:
            return

        if text.startswith("/"):
            parts   = text.split()
            cmd     = parts[0].lower().split("@")[0]
            args    = parts[1:]
            handler = self._handlers.get(cmd)
            if handler:
                try:
                    reply = handler(args, chat_id)
                    if reply and isinstance(reply, str):
                        self._bot.send_to(chat_id, reply)
                except Exception as exc:
                    logger.error(f"[TelegramBot] cmd {cmd}: {exc}")
            else:
                self._bot.send_to(chat_id,
                    f"⚠️ Unknown command: <code>{cmd}</code>\n"
                    "Send /menu to open the main menu.")
        else:
            # Plain text → conversation handler
            if self._text_handler:
                try:
                    reply = self._text_handler(text, chat_id)
                    if reply and isinstance(reply, str):
                        self._bot.send_to(chat_id, reply)
                except Exception as exc:
                    logger.error(f"[TelegramBot] text handler: {exc}")


# ── Singleton ─────────────────────────────────────────────────────────────────

_notifier: Optional[TelegramNotifier] = None

def get_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
