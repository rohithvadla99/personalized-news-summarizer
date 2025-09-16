from config import NEWS_API_KEY
import requests
import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "news.db")


def assign_topic(title, description):
    text = (title or "") + " " + (description or "")
    text = text.lower()
    if any(word in text for word in ["sport", "game", "football", "basketball", "tennis", "soccer"]):
        return "Sports"
    elif any(word in text for word in ["tech", "technology", "ai", "gadgets", "software", "app", "device"]):
        return "Tech"
    elif any(word in text for word in ["politic", "election", "government", "policy", "senate", "congress"]):
        return "Politics"
    elif any(word in text for word in ["business", "market", "stock", "finance", "economy", "trade"]):
        return "Business"
    else:
        return "Other"

# ----- Fetch news from NewsAPI -----
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    return articles

# ----- Store news into database -----
def store_news(articles):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for article in articles:
        title = article.get("title") or ""
        content = article.get("content") or ""
        description = article.get("description") or ""
        source = article.get("source", {}).get("name") or ""
        publishedAt = article.get("publishedAt") or ""
        summary = ""  # will be updated later
        sentiment = ""  # will be updated later
        topic = assign_topic(title, description)

        c.execute('''
            INSERT OR IGNORE INTO articles(
                title, content, description, source, publishedAt, summary, sentiment, topic
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (title, content, description, source, publishedAt, summary, sentiment, topic))

    conn.commit()
    conn.close()

# ----- Main -----
if __name__ == "__main__":
    news = fetch_news()
    store_news(news)
    print(f"Fetched and stored {len(news)} articles with topics")

# ----- Main -----
if __name__ == "__main__":
    news = fetch_news()
    store_news(news)
    print(f"Fetched and stored {len(news)} articles with topics")
