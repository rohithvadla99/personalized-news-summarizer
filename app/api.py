"""
Endpoints:
    GET  /                          health check
    GET  /articles                  list articles with filters
    GET  /articles/{id}             single article detail
    GET  /sentiment/summary         sentiment breakdown by topic
    GET  /sentiment/trend           daily sentiment trend
    GET  /sources                   source diversity stats
    GET  /metrics                   platform scale metrics
    POST /fetch                     trigger a news fetch (requires API key)
"""

import os
import sys
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fastapi             import FastAPI, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic            import BaseModel
from typing              import Optional
from datetime            import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_HERE, ".env"))
except ImportError:
    pass

from config  import DB_PATH, NEWS_API_KEY
from metrics import (
    sentiment_trend,
    source_breakdown,
    topic_sentiment_scores,
    reading_time_saved,
    db_stats,
    system_metrics,
)

app = FastAPI(
    title       = "NewsIQ API",
    description = "REST API for the NewsIQ news intelligence platform. "
                  "Exposes article summaries, sentiment scores, and platform metrics.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def verify_api_key(x_api_key: str = Header(...)):
    """
    Simple API key auth for write endpoints.
    Pass your NewsAPI key as the X-Api-Key header.
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-Api-Key header required")
    return x_api_key


class ArticleOut(BaseModel):
    id:          int
    title:       str
    summary:     str
    sentiment:   str
    topic:       str
    source:      str
    publishedAt: str
    url:         Optional[str] = None

class SentimentSummaryOut(BaseModel):
    topic:        str
    positive_pct: float
    negative_pct: float
    neutral_pct:  float
    net_score:    float

class MetricsOut(BaseModel):
    total_articles:  int
    unique_sources:  int
    total_tokens:    int
    total_words:     int
    total_users:     int
    avg_compression: float
    minutes_saved:   int
    fetch_runs:      int


@app.get("/", tags=["Health"])
def root():
    """Health check."""
    return {
        "status":  "ok",
        "service": "NewsIQ API",
        "version": "1.0.0",
        "docs":    "/docs",
    }


@app.get("/articles", response_model=list[ArticleOut], tags=["Articles"])
def list_articles(
    topic:     Optional[str] = Query(None, description="Filter by topic: Sports, Tech, Politics, Business"),
    sentiment: Optional[str] = Query(None, description="Filter by sentiment: POSITIVE, NEGATIVE, NEUTRAL"),
    search:    Optional[str] = Query(None, description="Search in title or summary"),
    limit:     int           = Query(20,  ge=1, le=100, description="Number of results (max 100)"),
    offset:    int           = Query(0,   ge=0,          description="Pagination offset"),
):
    """
    List articles with optional filters.

    Example:
        GET /articles?topic=Tech&sentiment=POSITIVE&limit=10
    """
    conn   = get_conn()
    c      = conn.cursor()
    query  = "SELECT id, title, summary, sentiment, topic, source, publishedAt, url FROM articles WHERE 1=1"
    params = []

    if topic:
        query += " AND topic=?";     params.append(topic)
    if sentiment:
        query += " AND sentiment=?"; params.append(sentiment.upper())
    if search:
        query += " AND (title LIKE ? OR summary LIKE ?)"; params.extend([f"%{search}%", f"%{search}%"])

    query += " ORDER BY publishedAt DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    return [
        ArticleOut(
            id          = r["id"],
            title       = r["title"]       or "",
            summary     = r["summary"]     or "",
            sentiment   = r["sentiment"]   or "NEUTRAL",
            topic       = r["topic"]       or "",
            source      = r["source"]      or "",
            publishedAt = r["publishedAt"] or "",
            url         = r["url"],
        )
        for r in rows
    ]


@app.get("/articles/{article_id}", response_model=ArticleOut, tags=["Articles"])
def get_article(article_id: int):
    """Get a single article by ID."""
    conn = get_conn()
    c    = conn.cursor()
    c.execute(
        "SELECT id, title, summary, sentiment, topic, source, publishedAt, url FROM articles WHERE id=?",
        (article_id,)
    )
    row = c.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found")

    return ArticleOut(
        id          = row["id"],
        title       = row["title"]       or "",
        summary     = row["summary"]     or "",
        sentiment   = row["sentiment"]   or "NEUTRAL",
        topic       = row["topic"]       or "",
        source      = row["source"]      or "",
        publishedAt = row["publishedAt"] or "",
        url         = row["url"],
    )


@app.get("/sentiment/summary", response_model=list[SentimentSummaryOut], tags=["Sentiment"])
def sentiment_summary():
    """
    Get sentiment breakdown (positive %, negative %, net score) per topic.
    Useful for building dashboards or trading signals.
    """
    df = topic_sentiment_scores()
    if df.empty:
        return []
    return [
        SentimentSummaryOut(
            topic        = row["topic"],
            positive_pct = row["positive_pct"],
            negative_pct = row["negative_pct"],
            neutral_pct  = row["neutral_pct"],
            net_score    = row["net_score"],
        )
        for _, row in df.iterrows()
    ]


@app.get("/sentiment/trend", tags=["Sentiment"])
def sentiment_trend_endpoint(
    days:  int           = Query(7,    ge=1, le=90,  description="Lookback window in days"),
    topic: Optional[str] = Query(None,               description="Filter by topic"),
):
    """
    Get daily sentiment counts for the last N days.
    Returns date-indexed POSITIVE / NEGATIVE / NEUTRAL counts.
    """
    df = sentiment_trend(days=days, topic=topic or "All")
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.get("/sources", tags=["Sources"])
def sources(
    topic: Optional[str] = Query(None, description="Filter by topic"),
    limit: int           = Query(10,   ge=1, le=50),
):
    """Get article counts per news source."""
    df = source_breakdown(topic=topic or "All", limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@app.get("/metrics", response_model=MetricsOut, tags=["Metrics"])
def platform_metrics():
    """
    Get platform-wide scale metrics.
    Useful for monitoring token throughput, user growth, and compression ratios.
    """
    m = system_metrics()
    return MetricsOut(
        total_articles  = m["total_articles"],
        unique_sources  = m["unique_sources"],
        total_tokens    = m["total_tokens"],
        total_words     = m["total_words"],
        total_users     = m["total_users"],
        avg_compression = m["avg_compression"],
        minutes_saved   = m["minutes_saved"],
        fetch_runs      = m["total_runs"],
    )


@app.post("/fetch", tags=["Pipeline"])
def trigger_fetch(api_key: str = Depends(verify_api_key)):
    """
    Trigger a news fetch run.
    Requires X-Api-Key header with your NewsAPI key.

    This runs the full pipeline:
    - Fetch from NewsAPI (4 categories)
    - Async-scrape full article text
    - Summarise with BART
    - Score sentiment with DistilBERT
    - Store in DB
    """
    try:
        from fetch_news import fetch_and_store
        total, new, duration = fetch_and_store(api_key)
        return {
            "status":           "ok",
            "articles_fetched": total,
            "articles_new":     new,
            "duration_sec":     duration,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sentiment/score", tags=["Sentiment"])
def live_sentiment_score(text: str = Query(..., description="Text to score")):
    """
    Score the sentiment of any text string on-the-fly.
    Returns label (POSITIVE/NEGATIVE/NEUTRAL) and confidence score.

    Example:
        GET /sentiment/score?text=Apple+reports+record+earnings
    """
    try:
        from transformers import pipeline as hf_pipeline
        analyzer = hf_pipeline("sentiment-analysis")
        truncated = " ".join(text.split()[:400])
        result    = analyzer(truncated)[0]
        return {
            "text":       text[:200],
            "label":      result["label"].upper(),
            "confidence": round(result["score"], 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
