“””
MQFS - Mediterranean Quantitative Finance Society
SentimentProvider: FinBERT + RSS News Integration

Fetches financial news headlines from public RSS feeds and scores them with
FinBERT (ProsusAI/finbert), a BERT model fine-tuned on financial text.

NEWS SOURCES (no API key required):
Primary:  Yahoo Finance RSS feeds
URL: https://feeds.finance.yahoo.com/rss/2.0/headline?s=TICKER
Provides historical articles, higher volume than yfinance.news.
Fallback: yfinance .news property (recent ~30-100 articles only).

```
RSS advantages over yfinance.news:
    - Historical coverage: articles retrievable for any date range
    - Higher volume: typically 50-200 per feed fetch
    - Standard format: RFC 2822 pubDate, easy to parse
```

SENTIMENT SCORE CONVENTION:
+1.0  =  strongly positive
0.0  =  neutral / no news
-1.0  =  strongly negative
Formula: score = P(positive) - P(negative)

FALLBACK CHAIN:
1. FinBERT (ProsusAI/finbert) via HuggingFace transformers  [best accuracy]
2. VADER (vaderSentiment) if transformers/torch unavailable  [lightweight]

Author:   Nicolo Angileri (MQFS)
Version:  3.1
“””

import logging
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np
import pandas as pd

logger = logging.getLogger(**name**)

# ==================== RSS CONFIG ====================

# Yahoo Finance RSS - public, no API key required

YAHOO_RSS_URL = (
“https://feeds.finance.yahoo.com/rss/2.0/headline”
“?s={ticker}&region=US&lang=en-US”
)

_HTTP_HEADERS = {
“User-Agent”: (
“Mozilla/5.0 (compatible; MQFS-ResearchBot/3.1; “
“+https://github.com/nicoloangileri/MQFS-Semantic-Arbitrage)”
)
}

# ==================== SENTIMENT PROVIDER ====================

class SentimentProvider:
“””
Daily FinBERT sentiment for a pair of financial assets.

```
News is fetched from Yahoo Finance RSS (free, no key, historical coverage),
falling back to yfinance .news when RSS is unavailable.

Usage:
```python
provider = SentimentProvider(use_finbert=True)
sentiment_vec = provider.build_daily_sentiment(
    ticker_x="KO",
    ticker_y="PEP",
    date_index=price_series.index,
)
```
"""

FINBERT_MODEL      = "ProsusAI/finbert"
MAX_HEADLINE_CHARS = 512
RSS_TIMEOUT_SEC    = 15
RSS_MAX_RETRIES    = 2

def __init__(self, use_finbert: bool = True, device: int = -1) -> None:
    """
    Args:
        use_finbert: Load FinBERT (requires transformers + torch).
                     Auto-falls back to VADER on ImportError.
        device:      -1 = CPU, 0 = first GPU.

    Raises:
        ImportError: If neither FinBERT nor VADER are installed.
    """
    self.use_finbert = use_finbert
    self._pipeline   = None
    self._vader      = None

    if use_finbert:
        self._try_load_finbert(device)

    if not self.use_finbert:
        self._try_load_vader()

# ------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------

def _try_load_finbert(self, device: int) -> None:
    try:
        from transformers import pipeline as hf_pipeline
        # top_k=None returns all labels: positive/negative/neutral
        # Requires transformers >= 4.30.0
        self._pipeline = hf_pipeline(
            "text-classification",
            model=self.FINBERT_MODEL,
            tokenizer=self.FINBERT_MODEL,
            device=device,
            top_k=None,
        )
        logger.info(f"FinBERT loaded: {self.FINBERT_MODEL} (device={device})")
    except Exception as exc:
        logger.warning(f"FinBERT unavailable ({exc}). Falling back to VADER.")
        self.use_finbert = False

def _try_load_vader(self) -> None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()
        logger.info("VADER loaded as sentiment fallback.")
    except ImportError as exc:
        raise ImportError(
            "No sentiment provider available.\n"
            "Install FinBERT: pip install transformers torch\n"
            "Or VADER:        pip install vaderSentiment"
        ) from exc

# ------------------------------------------------------------------
# SCORING
# ------------------------------------------------------------------

def _score_finbert(self, headline: str) -> float:
    """Returns P(positive) - P(negative) in [-1.0, +1.0]."""
    truncated    = headline[:self.MAX_HEADLINE_CHARS]
    label_scores = self._pipeline(truncated)[0]   # List[Dict] with top_k=None
    score_map    = {item['label'].lower(): item['score'] for item in label_scores}
    return score_map.get('positive', 0.0) - score_map.get('negative', 0.0)

def _score_vader(self, headline: str) -> float:
    """Returns VADER compound score in [-1.0, +1.0]."""
    return self._vader.polarity_scores(headline)['compound']

def score_headlines(self, headlines: List[str]) -> List[float]:
    """
    Scores a list of headlines. Per-article errors return 0.0.

    Args:
        headlines: Text headlines.

    Returns:
        List[float] in [-1.0, +1.0], same length as input.
    """
    if not headlines:
        return []

    scores: List[float] = []
    for headline in headlines:
        if not headline or not str(headline).strip():
            scores.append(0.0)
            continue
        try:
            if self.use_finbert:
                scores.append(float(self._score_finbert(str(headline))))
            else:
                scores.append(float(self._score_vader(str(headline))))
        except Exception as exc:
            logger.warning(f"Scoring error '{str(headline)[:60]}': {exc}")
            scores.append(0.0)

    return scores

# ------------------------------------------------------------------
# RSS FETCH (primary source)
# ------------------------------------------------------------------

def _fetch_rss(self, url: str) -> List[Dict]:
    """
    Fetches and parses a Yahoo Finance RSS feed.

    Handles network errors, malformed XML, and missing fields gracefully.
    Retries up to RSS_MAX_RETRIES times on transient network errors.

    Args:
        url: Full RSS feed URL.

    Returns:
        List of dicts {'title': str, 'pubDate': pd.Timestamp | None}.
        Returns [] on unrecoverable error.
    """
    raw_xml = None

    for attempt in range(self.RSS_MAX_RETRIES + 1):
        try:
            req      = Request(url, headers=_HTTP_HEADERS)
            response = urlopen(req, timeout=self.RSS_TIMEOUT_SEC)
            raw_xml  = response.read()
            break
        except URLError as exc:
            if attempt < self.RSS_MAX_RETRIES:
                logger.debug(
                    f"RSS attempt {attempt + 1}/{self.RSS_MAX_RETRIES + 1} "
                    f"failed: {exc}. Retrying in 1.5s..."
                )
                time.sleep(1.5)
            else:
                logger.warning(
                    f"RSS feed unavailable after {self.RSS_MAX_RETRIES + 1} "
                    f"attempts: {exc}"
                )
                return []
        except Exception as exc:
            logger.warning(f"Unexpected RSS error ({url}): {exc}")
            return []

    if raw_xml is None:
        return []

    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as exc:
        logger.warning(f"RSS XML parse error: {exc}")
        return []

    articles: List[Dict] = []

    # RSS 2.0 structure: <rss><channel><item>...</item></channel></rss>
    for item in root.iter('item'):
        title_el   = item.find('title')
        pubdate_el = item.find('pubDate')

        title = (
            title_el.text.strip()
            if (title_el is not None and title_el.text)
            else None
        )
        if not title:
            continue

        pub_ts: Optional[pd.Timestamp] = None
        if pubdate_el is not None and pubdate_el.text:
            try:
                # RSS pubDate is RFC 2822: "Mon, 10 Mar 2026 14:30:00 +0000"
                dt     = parsedate_to_datetime(pubdate_el.text.strip())
                pub_ts = pd.Timestamp(dt).tz_localize(None).normalize()
            except Exception:
                try:
                    # Fallback: attempt ISO 8601 parse
                    pub_ts = (
                        pd.Timestamp(pubdate_el.text.strip())
                        .tz_localize(None)
                        .normalize()
                    )
                except Exception:
                    pub_ts = None

        articles.append({'title': title, 'pubDate': pub_ts})

    return articles

# ------------------------------------------------------------------
# YFINANCE FALLBACK
# ------------------------------------------------------------------

def _fetch_yfinance_news(self, ticker: str) -> List[Dict]:
    """
    Fetches news via yfinance as fallback when RSS is unavailable.

    Returns recent articles only (~30-100). For long backtests, coverage
    of historical dates will be limited.

    Args:
        ticker: Yahoo Finance ticker symbol.

    Returns:
        List of dicts {'title': str, 'pubDate': pd.Timestamp | None}.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    articles: List[Dict] = []
    try:
        news = yf.Ticker(ticker).news or []
        for article in news:
            title = (
                article.get('content', {}).get('title') or
                article.get('title', '')
            )
            timestamp = (
                article.get('content', {}).get('pubDate') or
                article.get('providerPublishTime')
            )

            if not title:
                continue

            pub_ts: Optional[pd.Timestamp] = None
            if timestamp is not None:
                try:
                    if isinstance(timestamp, (int, float)):
                        pub_ts = pd.Timestamp(timestamp, unit='s').normalize()
                    else:
                        pub_ts = (
                            pd.Timestamp(str(timestamp))
                            .tz_localize(None)
                            .normalize()
                        )
                except Exception:
                    pub_ts = None

            articles.append({'title': str(title), 'pubDate': pub_ts})

    except Exception as exc:
        logger.warning(f"yfinance.news fallback failed for {ticker}: {exc}")

    return articles

# ------------------------------------------------------------------
# MAIN FETCH
# ------------------------------------------------------------------

def fetch_news_for_ticker(self, ticker: str) -> pd.DataFrame:
    """
    Fetches, scores, and returns all available news for a ticker.

    Source priority:
        1. Yahoo Finance RSS  (free, historical, no API key)
        2. yfinance .news     (recent only, ~30-100 articles)

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'KO', 'PEP').

    Returns:
        pd.DataFrame: columns ['date', 'title', 'sentiment_score'],
                      sorted ascending by date, tz-naive dates.
                      Empty DataFrame if no news found.
    """
    empty_df = pd.DataFrame(columns=['date', 'title', 'sentiment_score'])

    # Primary: RSS
    rss_url  = YAHOO_RSS_URL.format(ticker=ticker)
    articles = self._fetch_rss(rss_url)

    if articles:
        logger.info(f"{ticker}: {len(articles)} articles from RSS.")
    else:
        # Fallback: yfinance.news
        logger.info(f"{ticker}: RSS unavailable, trying yfinance.news fallback.")
        articles = self._fetch_yfinance_news(ticker)

    if not articles:
        logger.warning(f"{ticker}: no news articles from any source.")
        return empty_df

    # Drop articles without a parseable date
    valid = [a for a in articles if a['pubDate'] is not None]
    if not valid:
        logger.warning(f"{ticker}: articles found but none with valid pubDate.")
        return empty_df

    titles = [a['title'] for a in valid]
    dates  = [a['pubDate'] for a in valid]
    scores = self.score_headlines(titles)

    result_df = (
        pd.DataFrame({
            'date':            dates,
            'title':           titles,
            'sentiment_score': [float(s) for s in scores],
        })
        .sort_values('date')
        .reset_index(drop=True)
    )

    logger.info(
        f"{ticker}: {len(result_df)} articles scored | "
        f"range [{result_df['date'].min().date()} -> {result_df['date'].max().date()}] | "
        f"mean sentiment = {result_df['sentiment_score'].mean():.3f}"
    )
    return result_df

# ------------------------------------------------------------------
# BUILD DAILY VECTOR
# ------------------------------------------------------------------

def build_daily_sentiment(
    self,
    ticker_x: str,
    ticker_y: str,
    date_index: pd.DatetimeIndex,
    fill_neutral: float = 0.0,
    forward_fill_days: int = 3,
) -> np.ndarray:
    """
    Builds a daily sentiment vector aligned to date_index.

    Algorithm:
        1. Average all articles published on each date (both tickers combined).
        2. Forward-fill for `forward_fill_days` days (news remains relevant).
        3. Remaining gaps filled with `fill_neutral` (default 0.0 = neutral).

    Coverage (real news vs. interpolated vs. neutral) is always logged.

    Args:
        ticker_x:          First Yahoo Finance ticker.
        ticker_y:          Second Yahoo Finance ticker.
        date_index:        DatetimeIndex of the price series.
        fill_neutral:      Sentiment value for days with no news (default 0.0).
        forward_fill_days: Max forward-fill window in days (default 3).

    Returns:
        np.ndarray of shape (len(date_index),), clipped to [-1.0, +1.0].
    """
    # Normalise to tz-naive midnight for alignment with news dates
    date_index_naive = pd.DatetimeIndex([
        ts.tz_localize(None).normalize() if ts.tzinfo is not None
        else ts.normalize()
        for ts in date_index
    ])

    # Fetch and merge news for both tickers
    df_x   = self.fetch_news_for_ticker(ticker_x)
    df_y   = self.fetch_news_for_ticker(ticker_y)
    df_all = pd.concat([df_x, df_y], ignore_index=True)

    if df_all.empty:
        logger.warning(
            "No news for either ticker. "
            "Returning neutral sentiment (0.0) for all days."
        )
        return np.full(len(date_index), fill_neutral, dtype=float)

    # Daily average, reindexed to price dates
    daily_avg = (
        df_all.groupby('date')['sentiment_score']
        .mean()
        .reindex(date_index_naive, fill_value=np.nan)
    )
    daily_avg.index = date_index  # restore original index

    # Forward-fill then fill remaining gaps
    daily_filled = daily_avg.ffill(limit=forward_fill_days).fillna(fill_neutral)
    result       = np.clip(daily_filled.values, -1.0, 1.0).astype(float)

    # Coverage report
    n_real  = int(
        df_all.groupby('date')['sentiment_score']
        .mean()
        .reindex(date_index_naive)
        .notna()
        .sum()
    )
    n_total = len(date_index)
    logger.info(
        f"Sentiment coverage: {n_real}/{n_total} days with real news "
        f"({100.0 * n_real / n_total:.1f}%) | "
        f"{n_total - n_real} days filled ({fill_neutral}) | "
        f"overall mean = {float(result.mean()):.4f}"
    )

    return result
```
