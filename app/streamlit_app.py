import streamlit as st
import sqlite3
import os
import requests
import re
from transformers import pipeline

# ─────────────────────────────────────────────
# Config  (single source of truth)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "news.db")

TOPICS = ["Sports", "Tech", "Politics", "Business"]

TOPIC_KEYWORDS = {
    "Tech":      ["AI", "software", "Apple", "Google", "chip", "tech", "startup", "cyber"],
    "Sports":    ["NBA", "NFL", "match", "tournament", "score", "athlete", "league", "cup"],
    "Politics":  ["president", "congress", "election", "senate", "democrat", "republican", "bill", "vote"],
    "Business":  ["earnings", "market", "stock", "IPO", "revenue", "economy", "trade", "fed"],
}

# ─────────────────────────────────────────────
# Model loading  (cached — loaded once per session)
# ─────────────────────────────────────────────
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Strip HTML tags, NewsAPI truncation marker, and extra whitespace."""
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[\+\d+ chars?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_summarize(text: str, summarizer) -> str:
    """Summarize only when the text is long enough; fall back gracefully."""
    words = text.split()
    if len(words) < 40:
        return text  # too short to meaningfully summarize
    min_len = min(30, max(1, len(words) // 4))
    try:
        return summarizer(text, max_length=120, min_length=min_len, do_sample=False)[0]["summary_text"]
    except Exception:
        return text[:300]

def analyze_sentiment(text: str, sentiment_analyzer) -> str:
    """Return POSITIVE / NEGATIVE / NEUTRAL — always upper-case."""
    if not text:
        return "NEUTRAL"
    truncated = " ".join(text.split()[:400])  # keep under 512-token limit
    try:
        return sentiment_analyzer(truncated)[0]["label"].upper()
    except Exception:
        return "NEUTRAL"

def classify_topic(text: str) -> str:
    """Keyword-based topic classification."""
    lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw.lower() in lower for kw in keywords):
            return topic
    return "Other"

# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Fetch & store
# ─────────────────────────────────────────────
# NewsAPI category → our topic label
NEWSAPI_CATEGORIES = {
    "Sports":   "sports",
    "Tech":     "technology",
    "Politics": "general",
    "Business": "business",
}

def fetch_and_store(api_key: str) -> tuple[int, int]:
    """
    Fetch articles per topic category from NewsAPI so every article
    has a guaranteed topic assigned. Returns (total_fetched, new_inserted).
    """
    summarizer         = load_summarizer()
    sentiment_analyzer = load_sentiment()
    conn               = get_connection()
    c                  = conn.cursor()
    total              = 0
    new_count          = 0

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
            st.warning(f"Could not fetch {topic} news: {e}")
            continue

        if data.get("status") != "ok":
            st.warning(f"NewsAPI error for {topic}: {data.get('message')}")
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
            summary   = safe_summarize(content, summarizer)
            sentiment = analyze_sentiment(content, sentiment_analyzer)

            c.execute('''
                INSERT OR IGNORE INTO articles
                    (title, content, description, source, publishedAt, summary, sentiment, topic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (title, content, description, source, published, summary, sentiment, topic))

            if c.rowcount > 0:
                new_count += 1

    conn.commit()
    conn.close()
    return total, new_count

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")
st.title("📰 Personalized News Summarizer")

# --- API key (from Streamlit secrets or sidebar input) ---
try:
    api_key = st.secrets.get("NEWS_API_KEY", "")
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("NEWS_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("NewsAPI key", type="password",
                                     help="Enter your key from newsapi.org")

# --- Fetch button (nothing runs until the user clicks) ---
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Fetch latest news", disabled=not api_key):
    if not api_key:
        st.sidebar.error("Please enter a NewsAPI key first.")
    else:
        with st.spinner("Fetching articles and running NLP models… (this takes ~30 s on first run)"):
            try:
                total, added = fetch_and_store(api_key)
                st.sidebar.success(f"Fetched {total} articles — {added} new.")
            except RuntimeError as e:
                st.sidebar.error(str(e))

# --- Display ---
conn = get_connection()
c    = conn.cursor()

c.execute("SELECT DISTINCT topic FROM articles ORDER BY topic")
available_topics = [row[0] for row in c.fetchall()]

if not available_topics:
    st.info("No articles yet. Use the sidebar to fetch the latest news.")
    conn.close()
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    selected_topic = st.radio("Topic", ["All"] + available_topics)

    st.markdown("---")
    sentiment_filter = st.multiselect(
        "Sentiment",
        ["POSITIVE", "NEGATIVE", "NEUTRAL"],
        default=["POSITIVE", "NEGATIVE", "NEUTRAL"],
    )

with col2:
    if selected_topic == "All":
        placeholders = ",".join("?" * len(sentiment_filter))
        c.execute(
            f"SELECT title, summary, sentiment, publishedAt, source FROM articles "
            f"WHERE sentiment IN ({placeholders}) ORDER BY publishedAt DESC",
            sentiment_filter,
        )
    else:
        placeholders = ",".join("?" * len(sentiment_filter))
        c.execute(
            f"SELECT title, summary, sentiment, publishedAt, source FROM articles "
            f"WHERE topic=? AND sentiment IN ({placeholders}) ORDER BY publishedAt DESC",
            [selected_topic] + sentiment_filter,
        )

    rows = c.fetchall()

    if not rows:
        st.warning("No articles match the current filters.")
    else:
        # Sentiment colour map
        COLOURS = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🔵"}
        st.markdown(f"**{len(rows)} article{'s' if len(rows) != 1 else ''}**")
        for title, summary, sentiment, published_at, source in rows:
            icon = COLOURS.get(sentiment, "⚪")
            with st.expander(f"{icon} {title}"):
                st.write(summary or "_No summary available._")
                st.caption(
                    f"**Source:** {source} &nbsp;|&nbsp; "
                    f"**Sentiment:** {sentiment} &nbsp;|&nbsp; "
                    f"**Published:** {published_at}"
                )

conn.close()