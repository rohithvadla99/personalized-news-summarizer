import sqlite3
import os
import requests

from config     import DB_PATH, NEWS_API_KEY, TOPICS
from preprocess import clean_text
from summarize  import summarize
from sentiment  import analyze_sentiment

# NewsAPI category names that map to our TOPICS list
NEWSAPI_CATEGORIES = {
    "Sports":   "sports",
    "Tech":     "technology",
    "Politics": "general",
    "Business": "business",
}


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT,
            content     TEXT,
            description TEXT,
            source      TEXT,
            publishedAt TEXT,
            summary     TEXT,
            sentiment   TEXT,
            topic       TEXT,
            UNIQUE(title, source)
        )
    ''')
    conn.commit()
    return conn


def fetch_and_store(api_key: str = NEWS_API_KEY) -> tuple:
    """
    Fetch articles from NewsAPI for each topic category separately,
    so every article is guaranteed to have a correct topic assigned.
    Returns (total_fetched, new_inserted).
    """
    if not api_key:
        raise ValueError(
            "NEWS_API_KEY is not set. Add it to .streamlit/secrets.toml."
        )

    conn      = get_connection()
    c         = conn.cursor()
    total     = 0
    new_count = 0

    for topic, category in NEWSAPI_CATEGORIES.items():
        url = (
            f"https://newsapi.org/v2/top-headlines"
            f"?country=us&category={category}&pageSize=10&apiKey={api_key}"
        )

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"Network error fetching {topic}: {e}")
            continue

        if data.get("status") != "ok":
            print(f"NewsAPI error for {topic}: {data.get('message')}")
            continue

        articles = data.get("articles", [])
        total += len(articles)

        for article in articles:
            title       = article.get("title") or ""
            raw_content = article.get("content") or article.get("description") or ""
            description = article.get("description") or ""
            source      = article.get("source", {}).get("name") or ""
            published   = article.get("publishedAt") or ""

            content   = clean_text(raw_content)
            summary   = summarize(content)
            sentiment = analyze_sentiment(content)

            c.execute('''
                INSERT OR IGNORE INTO articles
                    (title, content, description, source, publishedAt, summary, sentiment, topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (title, content, description, source, published, summary, sentiment, topic))

            if c.rowcount > 0:
                new_count += 1

    conn.commit()
    conn.close()
    print(f"Fetched {total} articles across all topics, {new_count} new added.")
    return total, new_count


if __name__ == "__main__":
    fetch_and_store()