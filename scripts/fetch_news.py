import sqlite3
import os
import requests
from transformers import pipeline

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # set in Streamlit secrets or locally

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
response = requests.get(url).json()
articles = response.get("articles", [])

summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

for article in articles:
    title = article.get("title", "")
    content = article.get("content") or article.get("description") or ""
    description = article.get("description", "")
    source = article.get("source", {}).get("name", "")
    publishedAt = article.get("publishedAt", "")
    topic = "Other"  # fallback topic

    if content:
        summary = summarizer(content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(content)[0]['label']
    else:
        summary = ""
        sentiment = "Neutral"

    # Insert while ignoring duplicates
    c.execute('''
        INSERT OR IGNORE INTO articles(title, content, description, source, publishedAt, summary, sentiment, topic)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, content, description, source, publishedAt, summary, sentiment, topic))

conn.commit()
conn.close()
print(f"Fetched and stored {len(articles)} articles (duplicates ignored).")
