"""
metrics.py — business metric queries for the NewsIQ dashboard.

All functions return data structures ready to pass directly into
Streamlit charts (dicts, lists, DataFrames).
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from config import DB_PATH


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


# ── Sentiment trend (last N days) ─────────────────────────────────────────
def sentiment_trend(days: int = 7, topic: str = "All") -> pd.DataFrame:
    """
    Returns daily sentiment counts for the last `days` days.
    Columns: date, POSITIVE, NEGATIVE, NEUTRAL
    """
    conn = _conn()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()

    query = '''
        SELECT
            DATE(publishedAt) as date,
            sentiment,
            COUNT(*) as cnt
        FROM articles
        WHERE publishedAt >= ?
    '''
    params = [since]
    if topic != "All":
        query += " AND topic = ?"
        params.append(topic)
    query += " GROUP BY date, sentiment ORDER BY date"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["date", "POSITIVE", "NEGATIVE", "NEUTRAL"])

    pivot = df.pivot_table(index="date", columns="sentiment", values="cnt", fill_value=0)
    for col in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot.reset_index()


# ── Source diversity ───────────────────────────────────────────────────────
def source_breakdown(topic: str = "All", limit: int = 10) -> pd.DataFrame:
    """
    Returns article counts per source, optionally filtered by topic.
    Columns: source, count
    """
    conn = _conn()
    if topic == "All":
        df = pd.read_sql_query('''
            SELECT source, COUNT(*) as count
            FROM articles
            WHERE source != ''
            GROUP BY source
            ORDER BY count DESC
            LIMIT ?
        ''', conn, params=[limit])
    else:
        df = pd.read_sql_query('''
            SELECT source, COUNT(*) as count
            FROM articles
            WHERE topic = ? AND source != ''
            GROUP BY source
            ORDER BY count DESC
            LIMIT ?
        ''', conn, params=[topic, limit])
    conn.close()
    return df


# ── Sentiment score per topic ──────────────────────────────────────────────
def topic_sentiment_scores() -> pd.DataFrame:
    """
    Returns a net sentiment score per topic:
      score = (POSITIVE - NEGATIVE) / total  * 100
    Columns: topic, positive_pct, negative_pct, neutral_pct, net_score
    """
    conn = _conn()
    df = pd.read_sql_query('''
        SELECT topic, sentiment, COUNT(*) as cnt
        FROM articles
        GROUP BY topic, sentiment
    ''', conn)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot_table(index="topic", columns="sentiment", values="cnt", fill_value=0)
    for col in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total"]        = pivot["POSITIVE"] + pivot["NEGATIVE"] + pivot["NEUTRAL"]
    pivot["positive_pct"] = (pivot["POSITIVE"] / pivot["total"] * 100).round(1)
    pivot["negative_pct"] = (pivot["NEGATIVE"] / pivot["total"] * 100).round(1)
    pivot["neutral_pct"]  = (pivot["NEUTRAL"]  / pivot["total"] * 100).round(1)
    pivot["net_score"]    = (pivot["positive_pct"] - pivot["negative_pct"]).round(1)

    return pivot.reset_index()[["topic", "positive_pct", "negative_pct", "neutral_pct", "net_score"]]


# ── Reading time saved ─────────────────────────────────────────────────────
def reading_time_saved() -> dict:
    """
    Computes aggregate reading time saved across all articles.
    Assumes average reading speed of 200 words/min.
    """
    conn = _conn()
    c = conn.cursor()
    c.execute('''
        SELECT
            SUM(LENGTH(content) - LENGTH(summary))  as chars_saved,
            COUNT(*) as total_articles,
            SUM(LENGTH(content))  as total_chars,
            SUM(LENGTH(summary))  as summary_chars
        FROM articles
        WHERE content != '' AND summary != ''
    ''')
    row = c.fetchone()
    conn.close()

    if not row or not row[0]:
        return {"articles": 0, "minutes_saved": 0, "compression_pct": 0}

    chars_saved, total, orig, summ = row
    words_saved     = chars_saved / 5       # avg 5 chars/word
    minutes_saved   = words_saved / 200     # 200 wpm reading speed
    compression_pct = (1 - summ / orig) * 100 if orig else 0

    return {
        "articles":        total,
        "minutes_saved":   round(minutes_saved),
        "compression_pct": round(compression_pct, 1),
    }


# ── Company mention tracker ────────────────────────────────────────────────
def company_mentions(companies: list[str]) -> pd.DataFrame:
    """
    Counts mentions of each company name in article titles + content.
    Returns DataFrame with columns: company, mentions, avg_sentiment_score
    where sentiment_score = +1 for POSITIVE, -1 for NEGATIVE, 0 for NEUTRAL.
    """
    if not companies:
        return pd.DataFrame(columns=["company", "mentions", "sentiment_score"])

    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT title, content, sentiment FROM articles")
    rows = c.fetchall()
    conn.close()

    results = []
    for company in companies:
        lower = company.lower()
        mentions = 0
        score_sum = 0
        for title, content, sentiment in rows:
            text = f"{title} {content}".lower()
            if lower in text:
                mentions += 1
                score_sum += {"POSITIVE": 1, "NEGATIVE": -1}.get(sentiment, 0)

        results.append({
            "company":         company,
            "mentions":        mentions,
            "sentiment_score": round(score_sum / mentions, 2) if mentions else 0,
        })

    return pd.DataFrame(results).sort_values("mentions", ascending=False)


# ── Volume over time ───────────────────────────────────────────────────────
def article_volume(days: int = 7) -> pd.DataFrame:
    """Returns daily article counts for the last N days."""
    conn = _conn()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query('''
        SELECT DATE(publishedAt) as date, topic, COUNT(*) as count
        FROM articles
        WHERE publishedAt >= ?
        GROUP BY date, topic
        ORDER BY date
    ''', conn, params=[since])
    conn.close()
    return df


# ── Overall DB stats ───────────────────────────────────────────────────────
def db_stats() -> dict:
    conn = _conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM articles")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT source) FROM articles")
    sources = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT topic) FROM articles")
    topics = c.fetchone()[0]
    c.execute("SELECT MAX(publishedAt) FROM articles")
    latest = c.fetchone()[0] or "—"
    conn.close()
    return {"total": total, "sources": sources, "topics": topics, "latest": latest}