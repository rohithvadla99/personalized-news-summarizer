import streamlit as st
import sqlite3
from config import DB_PATH

st.title("Personalized News Summarizer")

# --- Connect to database ---
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# --- Get all distinct topics from the database ---
c.execute("SELECT DISTINCT topic FROM articles")
topics_in_db = [row[0] for row in c.fetchall()]

# --- Dropdown for topics ---
topic = st.selectbox("Choose topic", topics_in_db)

# --- Fetch articles for selected topic ---
c.execute("SELECT title, summary, sentiment FROM articles WHERE topic=?", (topic,))
filtered_articles = c.fetchall()
conn.close()

# --- Debug info (optional) ---
st.write(f"Selected topic: {topic}")
st.write(f"Number of articles fetched: {len(filtered_articles)}")

# --- Display articles ---
if len(filtered_articles) == 0:
    st.write(f"No articles found for '{topic}' yet. Please fetch/update news first.")
else:
    for article in filtered_articles:
        title, summary, sentiment = article
        st.subheader(title)
        st.write(summary)
        st.write(f"Sentiment: {sentiment}")
        st.markdown("---")
