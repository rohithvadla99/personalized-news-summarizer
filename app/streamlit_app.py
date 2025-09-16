import streamlit as st
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")

st.title("Personalized News Summarizer")

# Connect to DB
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Get topics
c.execute("SELECT DISTINCT topic FROM articles")
topics = [row[0] for row in c.fetchall()]

if topics:
    topic = st.selectbox("Select a topic", topics)

    c.execute(
        "SELECT title, summary, sentiment, publishedAt FROM articles WHERE topic=? ORDER BY publishedAt DESC",
        (topic,)
    )
    rows = c.fetchall()

    if rows:
        for title, summary, sentiment, publishedAt in rows:
            st.subheader(title)
            st.write(summary)
            st.caption(f"Sentiment: {sentiment} | Published: {publishedAt}")
    else:
        st.warning(f"No articles found for '{topic}' yet. Please fetch/update news first.")
else:
    st.warning("No topics found. Please fetch/update news first.")

conn.close()
