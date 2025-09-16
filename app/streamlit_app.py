import streamlit as st
import sqlite3
import os
import requests
from transformers import pipeline
from datetime import datetime

# -----------------------
# Config
# -----------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")  # set in Streamlit secrets

st.title("Personalized News Summarizer")

# -----------------------
# Initialize DB
# -----------------------
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
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
# Fetch latest news
# -----------------------
st.info("Fetching latest news...")

url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
response = requests.get(url).json()
articles = response.get("articles", [])

summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

new_count = 0
for article in articles:
    title = article.get("title", "")
    content = article.get("content") or article.get("description") or ""
    description = article.get("description", "")
    source = article.get("source", {}).get("name", "")
    publishedAt = article.get("publishedAt", "")  # ISO format
    topic = "Other"  # fallback topic

    if content:
        summary = summarizer(content, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(content)[0]['label']
    else:
        summary = ""
        sentiment = "Neutral"

    # Insert new articles only; ignore duplicates
    try:
        c.execute('''
            INSERT OR IGNORE INTO articles(title, content, description, source, publishedAt, summary, sentiment, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (title, content, description, source, publishedAt, summary, sentiment, topic))
        if c.rowcount > 0:
            new_count += 1
    except Exception as e:
        st.error(f"Failed to insert article: {title} | {e}")

conn.commit()
st.success(f"Fetched {len(articles)} articles, {new_count} new added.")

# -----------------------
# Display articles
# -----------------------
c.execute("SELECT DISTINCT topic FROM articles")
topics = [row[0] for row in c.fetchall()]

if topics:
    topic = st.selectbox("Select a topic", topics)

    c.execute(
        "SELECT title, summary, sentiment, publishedAt, source FROM articles WHERE topic=? ORDER BY publishedAt DESC",
        (topic,)
    )
    rows = c.fetchall()

    if rows:
        for title, summary, sentiment, publishedAt, source in rows:
            st.subheader(title)
            st.write(summary)
            st.caption(f"Sentiment: {sentiment} | Published: {publishedAt} | Source: {source}")
    else:
        st.warning(f"No articles found for '{topic}' yet.")
else:
    st.warning("No topics found in the database.")

conn.close()
