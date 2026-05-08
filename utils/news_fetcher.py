"""
utils/news_fetcher.py
Free News & Economic Calendar Fetcher

Sources — all 100% free, no API key required:
─────────────────────────────────────────────
1. RSS Feeds   — Reuters, CNBC, MarketWatch, Yahoo Finance,
                 FXStreet, DailyFX, Investing.com, Nasdaq
                 pip install feedparser

2. ForexFactory — Economic calendar scrape (NFP, CPI, FOMC)
                 pip install requests beautifulsoup4

3. Yahoo Finance — index/stock headlines
                 pip install yfinance
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── optional imports (graceful degradation) ──────────────────────────────────
try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False
    logger.warning("[NewsFetcher] pip install feedparser")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False
    logger.warning("[NewsFetcher] pip install beautifulsoup4")

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False
    logger.warning("[NewsFetcher] pip install yfinance")


# ── RSS feed list — all free ─────────────────────────────────────────────────
RSS_FEEDS = {
    "Reuters Markets":    "https://feeds.reuters.com/reuters/businessNews",
    "CNBC Markets":       "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "CNBC Finance":       "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch":        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Yahoo Finance":      "https://finance.yahoo.com/news/rssindex",
    "FXStreet":           "https://www.fxstreet.com/rss/news",
    "DailyFX":            "https://www.dailyfx.com/feeds/all",
    "Investing.com":      "https://www.investing.com/rss/news.rss",
    "Nasdaq News":        "https://www.nasdaq.com/feed/rssoutbound?category=Markets",
}

# high-impact keywords found in headlines
HIGH_IMPACT_WORDS = {
    "fed", "fomc", "federal reserve", "powell", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "nfp", "non-farm", "payroll", "gdp",
    "recession", "unemployment", "jobs report", "interest rate",
    "ecb", "boe", "bank of england", "lagarde", "war", "sanctions",
    "tariff", "default", "nasdaq", "s&p 500", "earnings miss",
}


# ── Simple data classes ───────────────────────────────────────────────────────

class NewsItem:
    def __init__(self, title: str, source: str,
                 published: Optional[datetime] = None,
                 sentiment: str = "neutral",
                 impact: str = "low"):
        self.title     = title
        self.source    = source
        self.published = published or datetime.utcnow()
        self.sentiment = sentiment
        self.impact    = impact   # "high" | "medium" | "low"

    def __repr__(self):
        tag = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(self.impact, "⚪")
        return f"{tag} [{self.source}] {self.title[:90]}"


class CalendarEvent:
    def __init__(self, name: str, time_utc: datetime,
                 currency: str = "USD", impact: str = "medium",
                 forecast: str = "", previous: str = ""):
        self.name         = name
        self.time_utc     = time_utc
        self.currency     = currency
        self.impact       = impact
        self.forecast     = forecast
        self.previous     = previous
        self.minutes_away = int((time_utc - datetime.utcnow()).total_seconds() / 60)

    def __repr__(self):
        tag = "🚨" if self.impact == "high" else "⚠️"
        return (f"{tag} {self.currency} {self.name} "
                f"— {max(self.minutes_away, 0)} min "
                f"({self.time_utc.strftime('%H:%M')} UTC)")


# ── Main class ────────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    Free news & calendar fetcher.
    get_summary() → formatted text block ready for Rico's prompt.
    """

    CACHE_TTL = 300   # 5 minutes — avoid hammering the same source

    def __init__(self):
        self._cache:    Dict = {}
        self._cache_ts: Dict = {}

    # ── public ──────────────────────────────────────────────────────────────

    def get_headlines(self, symbols: Optional[List[str]] = None,
                      max_items: int = 12,
                      max_age_hours: int = 4) -> List[NewsItem]:
        """Return headlines from all sources, newest-first."""
        items: List[NewsItem] = []

        if _HAS_FEEDPARSER:
            items.extend(self._rss(max_age_hours))

        if _HAS_YFINANCE and symbols:
            items.extend(self._yfinance(symbols))

        # deduplicate
        seen, unique = set(), []
        for it in items:
            k = it.title[:55].lower()
            if k not in seen:
                seen.add(k)
                unique.append(it)

        unique.sort(key=lambda x: x.published, reverse=True)
        return unique[:max_items]

    def get_calendar_events(self, hours_ahead: int = 24,
                            min_impact: str = "medium") -> List[CalendarEvent]:
        """Return upcoming calendar events from ForexFactory, with static fallback."""
        events: List[CalendarEvent] = []

        if _HAS_REQUESTS and _HAS_BS4:
            events.extend(self._forexfactory())

        # static fallback (always works, even offline)
        events.extend(self._static_calendar())

        now    = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        rank   = {"high": 3, "medium": 2, "low": 1}
        min_r  = rank.get(min_impact, 2)

        filtered = [
            e for e in events
            if now <= e.time_utc <= cutoff and rank.get(e.impact, 1) >= min_r
        ]

        # deduplicate by name
        seen_names: set = set()
        unique_ev: List[CalendarEvent] = []
        for e in filtered:
            if e.name not in seen_names:
                seen_names.add(e.name)
                unique_ev.append(e)

        unique_ev.sort(key=lambda e: e.time_utc)
        return unique_ev

    def get_summary(self, symbols: Optional[List[str]] = None,
                    hours_ahead: int = 12) -> str:
        """Return a formatted text block ready for Rico's prompt."""
        headlines = self.get_headlines(symbols=symbols, max_items=8)
        events    = self.get_calendar_events(hours_ahead=hours_ahead)

        high_ev = [e for e in events if e.impact == "high"]
        med_ev  = [e for e in events if e.impact == "medium"]
        risk    = self._risk_level(headlines, high_ev)

        lines = ["═══ LIVE MARKET NEWS ═══"]

        # calendar
        if high_ev:
            lines.append(f"\n🚨 HIGH-IMPACT EVENTS (next {hours_ahead}h):")
            for e in high_ev[:5]:
                mins = max(e.minutes_away, 0)
                fc   = f" | forecast: {e.forecast}" if e.forecast else ""
                lines.append(f"  • {e.name} ({e.currency}) in {mins} min{fc}")
        else:
            lines.append(f"\n✅ No high-impact events in next {hours_ahead}h")

        if med_ev:
            lines.append("\n⚠️  Medium events:")
            for e in med_ev[:3]:
                lines.append(f"  • {e.name} ({e.currency}) in {max(e.minutes_away,0)} min")

        # headlines
        if headlines:
            lines.append("\n📰 Headlines:")
            for h in headlines:
                lines.append(f"  {h!r}")
        else:
            lines.append("\n📰 No headlines available right now.")

        lines.append(f"\n📊 Overall News Risk: {risk.upper()}")
        return "\n".join(lines)

    # ── RSS ──────────────────────────────────────────────────────────────────

    def _rss(self, max_age_hours: int = 4) -> List[NewsItem]:
        key = f"rss_{max_age_hours}"
        if self._cached(key):
            return self._cache[key]

        items:  List[NewsItem] = []
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:8]:
                    title = (entry.get("title") or "").strip()
                    if not title:
                        continue
                    raw = entry.get("published_parsed") or entry.get("updated_parsed")
                    pub = datetime(*raw[:6]) if raw else datetime.utcnow()
                    if pub < cutoff:
                        continue
                    items.append(NewsItem(
                        title=title, source=source,
                        published=pub,
                        sentiment=self._sentiment(title),
                        impact=self._impact(title),
                    ))
            except Exception as e:
                logger.debug(f"[RSS] {source}: {e}")

        self._save(key, items)
        return items

    # ── Yahoo Finance ────────────────────────────────────────────────────────

    def _yfinance(self, symbols: List[str]) -> List[NewsItem]:
        key = "yf_" + "_".join(symbols)
        if self._cached(key):
            return self._cache[key]

        yf_map = {
            "NQ": "NQ=F", "ES": "ES=F", "GOLD": "GC=F",
            "XAUUSD": "GC=F", "USTEC": "NQ=F", "USTECm": "NQ=F",
            "SPY": "SPY", "QQQ": "QQQ",
        }
        tickers = list({yf_map.get(s.upper(), s) for s in symbols})[:3]
        items:  List[NewsItem] = []

        for t in tickers:
            try:
                news = yf.Ticker(t).news or []
                for a in news[:5]:
                    title = (a.get("title") or "").strip()
                    if not title:
                        continue
                    pub = datetime.utcfromtimestamp(
                        a.get("providerPublishTime", time.time()))
                    items.append(NewsItem(
                        title=title,
                        source=f"Yahoo/{t}",
                        published=pub,
                        sentiment=self._sentiment(title),
                        impact=self._impact(title),
                    ))
            except Exception as e:
                logger.debug(f"[yfinance] {t}: {e}")

        self._save(key, items)
        return items

    # ── ForexFactory scrape ───────────────────────────────────────────────────

    def _forexfactory(self) -> List[CalendarEvent]:
        key = "ff_calendar"
        if self._cached(key):
            return self._cache[key]

        events: List[CalendarEvent] = []
        try:
            r = requests.get(
                "https://www.forexfactory.com/calendar",
                headers={"User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                )},
                timeout=10,
            )
            soup = BeautifulSoup(r.text, "html.parser")
            today = datetime.utcnow().date()
            cur_date = today

            for row in soup.select("tr.calendar__row"):
                # date cell
                dc = row.select_one("td.calendar__date span")
                if dc and dc.get_text(strip=True):
                    try:
                        cur_date = datetime.strptime(
                            dc.get_text(strip=True) + f" {today.year}",
                            "%a %b %d %Y").date()
                    except Exception:
                        pass

                ec = row.select_one("td.calendar__event span")
                if not ec:
                    continue

                name = ec.get_text(strip=True)
                cur  = (row.select_one("td.calendar__currency") or ec).get_text(strip=True)
                tc   = row.select_one("td.calendar__time")
                fc   = row.select_one("td.calendar__forecast")
                pc   = row.select_one("td.calendar__previous")
                ic   = row.select_one("td.calendar__impact span")

                impact = "low"
                if ic:
                    cls = " ".join(ic.get("class", []))
                    impact = "high" if "high" in cls else ("medium" if "medium" in cls else "low")

                ev_dt = datetime.combine(cur_date, datetime.min.time())
                if tc:
                    ts = tc.get_text(strip=True)
                    try:
                        tp = datetime.strptime(ts, "%I:%M%p")
                        ev_dt = datetime(cur_date.year, cur_date.month, cur_date.day,
                                         tp.hour, tp.minute) + timedelta(hours=5)
                    except Exception:
                        pass

                events.append(CalendarEvent(
                    name=name, time_utc=ev_dt, currency=cur, impact=impact,
                    forecast=(fc.get_text(strip=True) if fc else ""),
                    previous=(pc.get_text(strip=True) if pc else ""),
                ))
        except Exception as e:
            logger.warning(f"[ForexFactory] {e}")

        self._save(key, events)
        return events

    # ── Static calendar fallback ──────────────────────────────────────────────

    def _static_calendar(self) -> List[CalendarEvent]:
        """
        Recurring high-impact events — works offline, used as ForexFactory fallback.
        """
        now  = datetime.utcnow()
        week = now.weekday()   # 0=Mon … 6=Sun
        evs: List[CalendarEvent] = []

        # FOMC — Wednesday 18:00 UTC (approximate)
        if week == 2:
            evs.append(CalendarEvent(
                "FOMC Rate Decision (possible)",
                now.replace(hour=18, minute=0, second=0, microsecond=0),
                "USD", "high",
            ))

        # NFP — first Friday of the month, 12:30 UTC
        if week == 4 and 1 <= now.day <= 7:
            evs.append(CalendarEvent(
                "Non-Farm Payrolls",
                now.replace(hour=12, minute=30, second=0, microsecond=0),
                "USD", "high",
            ))

        # CPI — approx mid-month, Tue/Wed, 12:30 UTC
        if week in (1, 2) and 8 <= now.day <= 15:
            evs.append(CalendarEvent(
                "CPI Inflation Data (possible)",
                now.replace(hour=12, minute=30, second=0, microsecond=0),
                "USD", "high",
            ))

        # Jobless Claims — every Thursday 12:30 UTC
        if week == 3:
            evs.append(CalendarEvent(
                "Jobless Claims",
                now.replace(hour=12, minute=30, second=0, microsecond=0),
                "USD", "medium",
            ))

        return evs

    # ── NLP helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _sentiment(text: str) -> str:
        t = text.lower()
        neg = ["crash","plunge","fall","drop","decline","loss","recession",
               "fear","selloff","slump","rate hike","inflation","default","miss"]
        pos = ["surge","rally","rise","gain","record","beat","strong",
               "growth","recover","rate cut","stimulus","boom"]
        nb = sum(1 for w in neg if w in t)
        pb = sum(1 for w in pos if w in t)
        return "negative" if nb > pb else ("positive" if pb > nb else "neutral")

    @staticmethod
    def _impact(text: str) -> str:
        t = text.lower()
        if any(w in t for w in HIGH_IMPACT_WORDS):
            return "high"
        if any(w in t for w in ["market","stock","index","gold","oil","yield","dollar"]):
            return "medium"
        return "low"

    @staticmethod
    def _risk_level(headlines: List[NewsItem],
                    high_events: List[CalendarEvent]) -> str:
        if high_events:
            return "high"
        neg = sum(1 for h in headlines if h.sentiment == "negative")
        hi  = sum(1 for h in headlines if h.impact    == "high")
        if hi >= 2 or neg >= 3:
            return "high"
        if hi >= 1 or neg >= 2:
            return "medium"
        return "low"

    # ── cache ─────────────────────────────────────────────────────────────────

    def _cached(self, k: str) -> bool:
        return k in self._cache_ts and (time.time() - self._cache_ts[k]) < self.CACHE_TTL

    def _save(self, k: str, v) -> None:
        self._cache[k]    = v
        self._cache_ts[k] = time.time()


# ── singleton ─────────────────────────────────────────────────────────────────

_fetcher: Optional[NewsFetcher] = None

def get_fetcher() -> NewsFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = NewsFetcher()
    return _fetcher
