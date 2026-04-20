"""
PRISM Layer 0 — News Intelligence
Monitors: Tiingo news sentiment, ForexFactory economic calendar, geopolitical RSS feeds.

Outputs:
- news_bias: BULLISH / BEARISH / NEUTRAL per instrument
- event_flag: True if high-impact event is inside the configured blackout window
- risk_regime: RISK_ON / RISK_OFF / NEUTRAL (geopolitical)
- volume_anomaly: True if volume spike detected (future hook; currently always False)
"""
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import requests

logger = logging.getLogger(__name__)

TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")

# Event blackout window (in minutes) around high-impact events.
# Trades are blocked from [event_time - EVENT_BLACKOUT_BEFORE_MIN]
# to [event_time + EVENT_BLACKOUT_AFTER_MIN]. Overridable via env for tuning.
EVENT_BLACKOUT_BEFORE_MIN = int(os.environ.get("PRISM_EVENT_BLACKOUT_BEFORE_MIN", "30"))
EVENT_BLACKOUT_AFTER_MIN = int(os.environ.get("PRISM_EVENT_BLACKOUT_AFTER_MIN", "15"))

# Geopolitical keywords that shift gold to RISK-OFF (bullish for XAU, bearish for risk FX)
RISK_OFF_KEYWORDS = [
    "iran", "war", "conflict", "sanctions", "nuclear", "missile",
    "invasion", "attack", "military", "escalation", "crisis",
    "fed emergency", "recession", "default", "banking crisis",
    "tariff", "trade war", "supply shock",
]

RISK_ON_KEYWORDS = [
    "ceasefire", "peace deal", "rate cut", "stimulus", "recovery",
    "trade deal", "optimism", "gdp beat", "employment surge",
]

# ForexFactory high-impact event endpoint
FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Instrument to news ticker mapping
NEWS_TICKER_MAP = {
    "XAUUSD": ["GLD", "GC=F", "gold"],
    "EURUSD": ["EUR=X", "EURUSD"],
    "GBPUSD": ["GBP=X", "GBPUSD"],
    "USDJPY": ["JPY=X", "USDJPY"],
}


@dataclass
class NewsSignal:
    instrument: str
    timestamp: str
    news_bias: str           # BULLISH / BEARISH / NEUTRAL
    event_flag: bool         # True while inside [event - before, event + after]
    event_name: str          # Name of the triggering event if event_flag
    risk_regime: str         # RISK_ON / RISK_OFF / NEUTRAL
    sentiment_score: float   # -1.0 to 1.0
    geopolitical_active: bool
    sources: list
    volume_anomaly: bool = False  # Placeholder; wired in Phase 3 volume layer


def _extract_tiingo_sentiment(article: dict) -> Optional[float]:
    """
    Normalize Tiingo's sentiment field to a float in [-1, 1].

    Tiingo returns `sentiment` as either:
      * a float (older responses)
      * a dict like {"compound": -0.12, "pos": 0.0, "neg": 0.3, "neu": 0.7}
      * missing entirely
    Returns None if no usable value is found so the caller can fall back to
    keyword scoring.
    """
    s = article.get("sentiment") if isinstance(article, dict) else None
    if s is None:
        return None
    if isinstance(s, (int, float)):
        try:
            val = float(s)
        except (TypeError, ValueError):
            return None
        return max(-1.0, min(1.0, val))
    if isinstance(s, dict):
        # Prefer a compound score; fall back to pos - neg
        if "compound" in s:
            try:
                return max(-1.0, min(1.0, float(s["compound"])))
            except (TypeError, ValueError):
                return None
        try:
            pos = float(s.get("pos", 0.0) or 0.0)
            neg = float(s.get("neg", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        return max(-1.0, min(1.0, pos - neg))
    return None


class NewsIntelligence:
    """Layer 0: News and event monitoring for PRISM signal generation."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Token {TIINGO_KEY}",
        })

    def get_signal(self, instrument: str) -> NewsSignal:
        """Main entry: run all checks and return NewsSignal for instrument."""
        sentiment = self._get_tiingo_sentiment(instrument)
        event_flag, event_name = self._check_economic_calendar(instrument)
        risk_regime, geo_active = self._check_geopolitical(instrument)
        news_bias = self._derive_bias(sentiment, risk_regime, instrument)

        return NewsSignal(
            instrument=instrument,
            timestamp=datetime.now(timezone.utc).isoformat(),
            news_bias=news_bias,
            event_flag=event_flag,
            event_name=event_name,
            risk_regime=risk_regime,
            sentiment_score=sentiment,
            geopolitical_active=geo_active,
            sources=["tiingo", "forex_factory_calendar", "geopolitical_rss"],
            volume_anomaly=False,
        )

    def _get_tiingo_sentiment(self, instrument: str) -> float:
        """Pull last 4h Tiingo news for instrument, return average sentiment -1 to 1."""
        if not TIINGO_KEY:
            return 0.0
        tickers = NEWS_TICKER_MAP.get(instrument, [])
        if not tickers:
            return 0.0
        scores: List[float] = []
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
            for ticker in tickers[:2]:  # Limit to 2 tickers to avoid rate limits
                url = "https://api.tiingo.com/tiingo/news"
                params = {"tickers": ticker, "startDate": cutoff, "limit": 10}
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    continue
                articles = r.json()
                if not isinstance(articles, list):
                    continue
                for article in articles:
                    if not isinstance(article, dict):
                        continue
                    structured = _extract_tiingo_sentiment(article)
                    if structured is not None:
                        scores.append(structured)
                        continue
                    # No structured sentiment — fall back to keyword scoring
                    title = str(article.get("title", "")).lower()
                    scores.append(self._keyword_sentiment(title, instrument))
        except Exception as e:
            logger.warning(f"Tiingo news fetch failed for {instrument}: {e}")
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _keyword_sentiment(self, text: str, instrument: str) -> float:
        """Score text -1 to 1 based on keyword matching."""
        text = text.lower()
        positive = [
            "rally", "surge", "gain", "rise", "bullish", "buy", "strong",
            "beat", "optimism", "recovery", "deal", "ceasefire",
        ]
        negative = [
            "fall", "drop", "crash", "decline", "bearish", "sell", "weak",
            "miss", "recession", "war", "crisis", "sanctions", "tariff",
        ]
        # Gold-specific adjustments (risk-off = bullish for gold)
        if instrument == "XAUUSD":
            negative_gold = ["risk-on", "rate hike", "strong dollar", "yields rise"]
            positive_gold = ["risk-off", "safe haven", "inflation", "uncertainty", "crisis"]
            for kw in negative_gold:
                if kw in text:
                    return -0.5
            for kw in positive_gold:
                if kw in text:
                    return 0.5
        pos = sum(1 for w in positive if w in text)
        neg = sum(1 for w in negative if w in text)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _check_economic_calendar(self, instrument: str) -> Tuple[bool, str]:
        """
        Check ForexFactory calendar for high-impact events inside the blackout window.
        The blackout window is [event_time - EVENT_BLACKOUT_BEFORE_MIN,
        event_time + EVENT_BLACKOUT_AFTER_MIN]. Trades are blocked while `now`
        is within that window for a relevant currency.
        """
        currency_map = {
            "XAUUSD": ["USD"],
            "EURUSD": ["EUR", "USD"],
            "GBPUSD": ["GBP", "USD"],
            "USDJPY": ["USD", "JPY"],
        }
        relevant_currencies = currency_map.get(instrument, ["USD"])

        try:
            r = requests.get(FOREX_FACTORY_URL, timeout=10)
            if r.status_code != 200:
                return False, ""
            events = r.json()
            if not isinstance(events, list):
                return False, ""
            now = datetime.now(timezone.utc)

            for event in events:
                if not isinstance(event, dict):
                    continue
                if event.get("impact") != "High":
                    continue
                if event.get("currency") not in relevant_currencies:
                    continue
                event_time = _parse_event_time(event.get("date", ""))
                if event_time is None:
                    continue
                window_start = event_time - timedelta(minutes=EVENT_BLACKOUT_BEFORE_MIN)
                window_end = event_time + timedelta(minutes=EVENT_BLACKOUT_AFTER_MIN)
                if window_start <= now <= window_end:
                    return True, event.get("title") or event.get("event") or "High-impact event"
        except Exception as e:
            logger.warning(f"Economic calendar check failed: {e}")

        return False, ""

    def _check_geopolitical(self, instrument: str) -> Tuple[str, bool]:
        """
        Check RSS feeds for geopolitical keywords.
        Returns (risk_regime, geo_active).
        Headlines are deduped across feeds to avoid double-counting.
        """
        rss_feeds = [
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
        ]
        geo_relevant = ["XAUUSD", "USDJPY", "EURUSD", "GBPUSD"]
        if instrument not in geo_relevant:
            return "NEUTRAL", False

        try:
            import feedparser  # local import — optional dependency
        except ImportError:
            logger.warning(
                "feedparser not installed — skipping geopolitical RSS scan. "
                "Add `feedparser` to requirements.txt to enable."
            )
            return "NEUTRAL", False

        seen_titles: set = set()
        risk_off_count = 0
        risk_on_count = 0
        per_feed_counts: dict = {}

        for feed_url in rss_feeds:
            feed_off = 0
            feed_on = 0
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    title = (entry.get("title", "") or "").strip().lower()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    summary = (entry.get("summary", "") or "").lower()
                    text = f"{title} {summary}"
                    feed_off += sum(1 for kw in RISK_OFF_KEYWORDS if kw in text)
                    feed_on += sum(1 for kw in RISK_ON_KEYWORDS if kw in text)
            except Exception as e:
                logger.warning(f"RSS fetch failed for {feed_url}: {e}")
                continue
            risk_off_count += feed_off
            risk_on_count += feed_on
            per_feed_counts[feed_url] = (feed_off, feed_on)

        if per_feed_counts:
            logger.debug(f"Geopolitical RSS counts per feed: {per_feed_counts}")

        geo_active = risk_off_count > 2
        if risk_off_count > risk_on_count and risk_off_count > 2:
            return "RISK_OFF", geo_active
        if risk_on_count > risk_off_count and risk_on_count > 2:
            return "RISK_ON", geo_active
        return "NEUTRAL", geo_active

    def _derive_bias(self, sentiment: float, risk_regime: str, instrument: str) -> str:
        """Combine sentiment + geopolitical regime into directional bias."""
        if instrument == "XAUUSD":
            if risk_regime == "RISK_OFF":
                return "BULLISH"
            if risk_regime == "RISK_ON":
                return "BEARISH"
        elif instrument in ("EURUSD", "GBPUSD"):
            if risk_regime == "RISK_OFF":
                return "BEARISH"
            if risk_regime == "RISK_ON":
                return "BULLISH"
        if sentiment > 0.2:
            return "BULLISH"
        if sentiment < -0.2:
            return "BEARISH"
        return "NEUTRAL"

    def should_block_trade(self, signal: NewsSignal) -> Tuple[bool, str]:
        """Return (blocked, reason) if trade should be prevented."""
        if signal.event_flag:
            return True, f"High-impact event imminent: {signal.event_name}"
        return False, ""


def _parse_event_time(raw: str) -> Optional[datetime]:
    """
    Parse an ISO-8601 / ForexFactory-style datetime string to a UTC-aware datetime.
    Returns None if the input is empty or unparseable.
    """
    if not raw:
        return None
    try:
        # Handle trailing 'Z' (UTC designator), which fromisoformat doesn't accept pre-3.11
        cleaned = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
