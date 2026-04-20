"""
PRISM Layer 0 — News Intelligence
Monitors: Tiingo news sentiment, ForexFactory economic calendar, geopolitical RSS feeds.

Outputs:
- news_bias: BULLISH / BEARISH / NEUTRAL per instrument
- event_flag: True if high-impact event within 30 min (block trading)
- risk_regime: RISK_ON / RISK_OFF / NEUTRAL (geopolitical)
- volume_anomaly: True if volume spike detected
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import requests
import os

logger = logging.getLogger(__name__)

TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")

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
    event_flag: bool         # High-impact event imminent (block trading)
    event_name: str          # Name of upcoming event if event_flag
    risk_regime: str         # RISK_ON / RISK_OFF / NEUTRAL
    sentiment_score: float   # -1.0 to 1.0
    geopolitical_active: bool
    sources: list


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
            timestamp=datetime.utcnow().isoformat(),
            news_bias=news_bias,
            event_flag=event_flag,
            event_name=event_name,
            risk_regime=risk_regime,
            sentiment_score=sentiment,
            geopolitical_active=geo_active,
            sources=["tiingo", "forex_factory_calendar", "geopolitical_rss"],
        )

    def _get_tiingo_sentiment(self, instrument: str) -> float:
        """Pull last 4h Tiingo news for instrument, return average sentiment -1 to 1."""
        if not TIINGO_KEY:
            return 0.0
        tickers = NEWS_TICKER_MAP.get(instrument, [])
        if not tickers:
            return 0.0
        scores = []
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
            for ticker in tickers[:2]:  # Limit to 2 tickers to avoid rate limits
                url = "https://api.tiingo.com/tiingo/news"
                params = {"tickers": ticker, "startDate": cutoff, "limit": 10}
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code == 200:
                    articles = r.json()
                    for article in articles:
                        if "sentiment" in article:
                            scores.append(article["sentiment"])
                        else:
                            title = article.get("title", "").lower()
                            score = self._keyword_sentiment(title, instrument)
                            scores.append(score)
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

    def _check_economic_calendar(self, instrument: str) -> tuple:
        """
        Check ForexFactory calendar for high-impact events in next 30 minutes.
        Returns (event_flag, event_name).
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
            now = datetime.now(timezone.utc)
            window_end = now + timedelta(minutes=30)

            for event in events:
                if event.get("impact") != "High":
                    continue
                if event.get("currency") not in relevant_currencies:
                    continue
                try:
                    event_time_str = event.get("date", "")
                    if not event_time_str:
                        continue
                    event_time = datetime.fromisoformat(event_time_str.replace("Z", "+00:00"))
                    if now <= event_time <= window_end:
                        return True, event.get("title", "High-impact event")
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Economic calendar check failed: {e}")

        return False, ""

    def _check_geopolitical(self, instrument: str) -> tuple:
        """
        Check RSS feeds for geopolitical keywords.
        Returns (risk_regime, geo_active).
        risk_regime: RISK_OFF / RISK_ON / NEUTRAL
        """
        rss_feeds = [
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.bbci.co.uk/news/world/rss.xml",
        ]
        geo_relevant = ["XAUUSD", "USDJPY", "EURUSD", "GBPUSD"]
        if instrument not in geo_relevant:
            return "NEUTRAL", False

        risk_off_count = 0
        risk_on_count = 0

        for feed_url in rss_feeds:
            try:
                import feedparser
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:
                    text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                    risk_off_count += sum(1 for kw in RISK_OFF_KEYWORDS if kw in text)
                    risk_on_count += sum(1 for kw in RISK_ON_KEYWORDS if kw in text)
            except Exception as e:
                logger.warning(f"RSS fetch failed for {feed_url}: {e}")

        geo_active = risk_off_count > 2  # 3+ risk-off mentions = active
        if risk_off_count > risk_on_count and risk_off_count > 2:
            return "RISK_OFF", geo_active
        elif risk_on_count > risk_off_count and risk_on_count > 2:
            return "RISK_ON", geo_active
        return "NEUTRAL", geo_active

    def _derive_bias(self, sentiment: float, risk_regime: str, instrument: str) -> str:
        """Combine sentiment + geopolitical regime into directional bias."""
        # Gold: risk-off is bullish
        if instrument == "XAUUSD":
            if risk_regime == "RISK_OFF":
                return "BULLISH"
            elif risk_regime == "RISK_ON":
                return "BEARISH"
        # USD pairs: risk-off is bearish (USD strengthens vs risk FX)
        elif instrument in ["EURUSD", "GBPUSD"]:
            if risk_regime == "RISK_OFF":
                return "BEARISH"
            elif risk_regime == "RISK_ON":
                return "BULLISH"
        # Fall back to sentiment
        if sentiment > 0.2:
            return "BULLISH"
        elif sentiment < -0.2:
            return "BEARISH"
        return "NEUTRAL"

    def should_block_trade(self, signal: NewsSignal) -> tuple:
        """Return (blocked, reason) if trade should be prevented."""
        if signal.event_flag:
            return True, f"High-impact event imminent: {signal.event_name}"
        return False, ""
