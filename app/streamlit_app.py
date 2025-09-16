import streamlit as st
import sqlite3
import os
import requests
from transformers import pipeline

# -------------------------------
# Config / Environment variables
# -------------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")  # set this in Streamlit Secrets

# -------------------------------
# Initialize DB and table
# -------------------------------
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
        topic TEXT
    )
''')
conn.commit()

# -------------------------------
# Fetch news if DB is empty
# -------------------------------
c.execute("SELECT COUNT(*) FROM articles")
if c.fetchone()[0] == 0:
    st.info("Fetching news...")
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

        c.execute('''
            INSERT INTO articles(title, content, description, source, publishedAt, summary, sentiment, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (title, content, description, source, publishedAt, summary, sentiment, topic))

    conn.commit()
    st.success(f"Fetched and summarized {len(articles)} articles.")

# -------------------------------
# Display articles
# -------------------------------
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
    st.warning("No topics found. Please fetch news first.")

conn.close()
