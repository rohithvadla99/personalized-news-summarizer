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
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


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

# ── Market correlation ─────────────────────────────────────────────────────
def market_correlation(days: int = 30) -> dict:
    """
    Fetches S&P 500 (SPY), Tech (QQQ), and Finance (XLF) daily closing
    prices for the last N days and computes daily sentiment scores from
    the DB for the same period.

    Returns a dict with:
      - combined_df: DataFrame with date, sentiment_score, SPY, QQQ, XLF
      - correlations: dict of Pearson correlation coefficients per ticker
      - error: str if data could not be fetched (e.g. no internet)
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "yfinance not installed. Run: pip install yfinance"}

    from datetime import datetime, timedelta

    end   = datetime.utcnow()
    start = end - timedelta(days=days)

    TICKERS = {
        "S&P 500 (SPY)":    "SPY",
        "Tech (QQQ)":       "QQQ",
        "Finance (XLF)":    "XLF",
    }

    # ── Fetch market data ──────────────────────────────────────────────────
    try:
        market_frames = {}
        for label, ticker in TICKERS.items():
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                continue
            df.index = pd.to_datetime(df.index).date.astype(str)  # type: ignore
            # Handle MultiIndex columns from newer yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            market_frames[label] = df["Close"].rename(label)

        if not market_frames:
            return {"error": "Could not fetch market data. Check your internet connection."}

        market_df = pd.concat(market_frames.values(), axis=1).reset_index()
        market_df.rename(columns={"index": "date"}, inplace=True)

    except Exception as e:
        return {"error": f"Market data fetch failed: {e}"}

    # ── Fetch sentiment scores from DB ─────────────────────────────────────
    conn = _conn()
    sent_df = pd.read_sql_query("""
        SELECT
            DATE(publishedAt) as date,
            ROUND(
                (SUM(CASE WHEN sentiment='POSITIVE' THEN 1.0 ELSE 0 END) -
                 SUM(CASE WHEN sentiment='NEGATIVE' THEN 1.0 ELSE 0 END))
                / COUNT(*), 3
            ) as sentiment_score
        FROM articles
        WHERE publishedAt >= ?
        GROUP BY DATE(publishedAt)
        ORDER BY date
    """, conn, params=[start.isoformat()])
    conn.close()

    if sent_df.empty:
        return {"error": "Not enough article data. Fetch news across multiple days first."}

    # ── Merge on date ──────────────────────────────────────────────────────
    merged = pd.merge(sent_df, market_df, on="date", how="inner")

    if len(merged) < 3:
        return {"error": "Not enough overlapping days between news and market data. Fetch more news."}

    # ── Compute daily % change for market (more meaningful than raw price) ─
    for col in TICKERS.keys():
        if col in merged.columns:
            merged[f"{col} %"] = merged[col].pct_change() * 100

    # ── Pearson correlations ───────────────────────────────────────────────
    correlations = {}
    for col in TICKERS.keys():
        pct_col = f"{col} %"
        if pct_col in merged.columns:
            valid = merged[["sentiment_score", pct_col]].dropna()
            if len(valid) >= 3:
                corr = valid["sentiment_score"].corr(valid[pct_col])
                correlations[col] = round(corr, 3)

    return {
        "combined_df":  merged,
        "correlations": correlations,
        "tickers":      list(TICKERS.keys()),
        "error":        None,
    }


# ── System scale metrics ───────────────────────────────────────────────────
def system_metrics() -> dict:
    """
    Returns platform-wide scale metrics suitable for resume/about page.
    All numbers derived from real usage data in the DB.
    """
    conn = _conn()
    c = conn.cursor()

    # Total tokens processed across all articles
    c.execute("SELECT COALESCE(SUM(tokens_processed), 0) FROM articles")
    total_tokens = c.fetchone()[0]

    # Total fetch runs
    c.execute("SELECT COUNT(*) FROM fetch_log")
    total_runs = c.fetchone()[0]

    # Total articles ever processed
    c.execute("SELECT COUNT(*) FROM articles")
    total_articles = c.fetchone()[0]

    # Unique sources monitored
    c.execute("SELECT COUNT(DISTINCT source) FROM articles WHERE source != ''")
    unique_sources = c.fetchone()[0]

    # Daily active users (distinct users with activity in last 24h)
    c.execute("""
        SELECT COUNT(DISTINCT uid) FROM read_history
        WHERE read_at >= datetime('now', '-1 day')
    """)
    dau = c.fetchone()[0]

    # Weekly active users
    c.execute("""
        SELECT COUNT(DISTINCT uid) FROM read_history
        WHERE read_at >= datetime('now', '-7 days')
    """)
    wau = c.fetchone()[0]

    # Total registered users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    # Total articles read across all users
    c.execute("SELECT COUNT(*) FROM read_history")
    total_reads = c.fetchone()[0]

    # Average compression from fetch log
    c.execute("SELECT AVG(avg_compression) FROM fetch_log WHERE avg_compression > 0")
    avg_compression = c.fetchone()[0] or 0

    # Total words processed (rough: tokens * 0.75)
    total_words = int(total_tokens * 0.75)

    # Estimated reading time saved (avg article 800 words, 200wpm, compression applied)
    c.execute("""
        SELECT COALESCE(SUM(LENGTH(content) - LENGTH(summary)), 0)
        FROM articles WHERE content != '' AND summary != ''
    """)
    chars_saved = c.fetchone()[0]
    minutes_saved = round((chars_saved / 5) / 200)

    conn.close()

    return {
        "total_tokens":     total_tokens,
        "total_words":      total_words,
        "total_articles":   total_articles,
        "unique_sources":   unique_sources,
        "total_runs":       total_runs,
        "dau":              dau,
        "wau":              wau,
        "total_users":      total_users,
        "total_reads":      total_reads,
        "avg_compression":  round(avg_compression, 1),
        "minutes_saved":    minutes_saved,
    }


def daily_token_volume(days: int = 30) -> pd.DataFrame:
    """Returns daily token processing volume for the last N days."""
    conn = _conn()
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query("""
        SELECT DATE(publishedAt) as date,
               SUM(tokens_processed) as tokens,
               COUNT(*) as articles
        FROM articles
        WHERE publishedAt >= ?
        GROUP BY DATE(publishedAt)
        ORDER BY date
    """, conn, params=[since])
    conn.close()
    return df