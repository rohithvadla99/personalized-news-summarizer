import sqlite3
import os
import requests
from transformers import pipeline

# -----------------------
# Config
# -----------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")
# Use environment variable or Streamlit secrets
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or os.environ.get("NEWS_API_KEY")  
if not NEWS_API_KEY:
    raise ValueError("Please set NEWS_API_KEY as an environment variable or in Streamlit secrets")

# -----------------------
# Connect to DB
# -----------------------
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Ensure table exists with UNIQUE constraint
c.execute('''
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,
    description TEXT,
    source TEXT,
    publishedAt TEXT,
    summary TEXT,
    sentiment TEXT,
    topic TEXT,
    UNIQUE(title, source)
)
''')
conn.commit()

# -----------------------
# Fetch news
# -----------------------
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
response = requests.get(url).json()
articles = response.get("articles", [])

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    revision="a4f8f3e"
)
sentiment_analyzer = pipeline("sentiment-analysis")

new_count = 0
for article in articles:
    title = article.get("title", "")
    content = article.get("content") or article.get("description") or ""
    description = article.get("description", "")
    source = article.get("source", {}).get("name", "")
    publishedAt = article.get("publishedAt", "")
    topic = "Other"

    if content:
        summary = summarizer(content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(content)[0]['label']
    else:
        summary = ""
        sentiment = "Neutral"

    # Insert only if not duplicate
    c.execute('''
        INSERT OR IGNORE INTO articles(title, content, description, source, publishedAt, summary, sentiment, topic)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, content, description, source, publishedAt, summary, sentiment, topic))

    if c.rowcount > 0:
        new_count += 1

conn.commit()
conn.close()
print(f"Fetched {len(articles)} articles, {new_count} new added (duplicates ignored).")
