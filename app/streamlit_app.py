import streamlit as st
import sqlite3
import os
import requests
from transformers import pipeline

# -----------------------
# Config
# -----------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

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
# Fetch news function
# -----------------------
def fetch_and_store_news():
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
    return new_count

# -----------------------
# Streamlit UI
# -----------------------
st.title("Personalized News Summarizer")

if st.button("Fetch / Update News"):
    added = fetch_and_store_news()
    st.success(f"Fetched and stored {added} new articles!")

# Display topics
c.execute("SELECT DISTINCT topic FROM articles")
topics = [row[0] for row in c.fetchall()]
selected_topic = st.selectbox("Choose topic", topics or ["Other"])

# Display articles for selected topic
c.execute("SELECT title, summary, sentiment FROM articles WHERE topic=?", (selected_topic,))
rows = c.fetchall()
if not rows:
    st.warning(f"No articles found for '{selected_topic}' yet. Please fetch/update news first.")
else:
    for title, summary, sentiment in rows:
        st.subheader(title)
        st.write(summary)
        st.write(f"**Sentiment:** {sentiment}")

conn.close()
