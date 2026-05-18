import streamlit as st
import sqlite3
import os
import sys
import requests
import re
import json
import pandas as pd
from transformers import pipeline

# Allow imports from project root (config.py etc.)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from auth       import handle_callback, restore_session, is_logged_in, get_user, logout, render_login
from user_prefs import init_user_tables, upsert_user, get_prefs, save_prefs, mark_read, get_read_ids, get_user_stats
from metrics    import sentiment_trend, source_breakdown, topic_sentiment_scores, reading_time_saved, company_mentions, db_stats

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NewsIQ",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "news.db")

ALL_TOPICS     = ["Sports", "Tech", "Politics", "Business"]
ALL_SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

NEWSAPI_CATEGORIES = {
    "Sports":   "sports",
    "Tech":     "technology",
    "Politics": "general",
    "Business": "business",
}

SENTIMENT_ICON = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🔵"}

# ─────────────────────────────────────────────
# Init DB user tables
# ─────────────────────────────────────────────
init_user_tables()

# ─────────────────────────────────────────────
# Google OAuth — handle redirect callback
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .value {
    font-size: 32px;
    font-weight: 600;
    color: #0f172a;
    font-family: 'DM Serif Display', serif;
    line-height: 1;
}
.metric-card .label {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-top: 6px;
}
.article-card {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color .15s;
}
.article-card:hover { border-color: #6366f1; }
.user-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #f1f5f9;
    border-radius: 20px;
    padding: 4px 12px 4px 4px;
    font-size: 13px;
}
.pill-avatar {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    object-fit: cover;
}
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
}
.tag-pos { background: #dcfce7; color: #166534; }
.tag-neg { background: #fee2e2; color: #991b1b; }
.tag-neu { background: #dbeafe; color: #1e40af; }
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #0f172a;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

# ─────────────────────────────────────────────
# NLP helpers
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[\+\d+ chars?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_summarize(text: str, summarizer) -> str:
    words = text.split()
    if len(words) < 40:
        return text
    if len(words) > 600:
        text = " ".join(words[:600])
        words = words[:600]
    min_len = min(30, max(1, len(words) // 4))
    try:
        return summarizer(text, max_length=120, min_length=min_len, do_sample=False)[0]["summary_text"]
    except Exception:
        return text[:300]

def analyze_sentiment(text: str, sentiment_analyzer) -> str:
    if not text:
        return "NEUTRAL"
    truncated = " ".join(text.split()[:400])
    try:
        return sentiment_analyzer(truncated)[0]["label"].upper()
    except Exception:
        return "NEUTRAL"

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
def fetch_and_store(api_key: str) -> tuple[int, int]:
    summarizer_model   = load_summarizer()
    sentiment_model    = load_sentiment()
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
            st.warning(f"Could not fetch {topic}: {e}")
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
            summary   = safe_summarize(content, summarizer_model)
            sentiment = analyze_sentiment(content, sentiment_model)

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
# LOGIN GATE
# ─────────────────────────────────────────────
handle_callback()    # exchanges ?code= for user info if present
restore_session()    # restores session from cookie if page was refreshed

if not is_logged_in():
    render_login()
    st.stop()

# On first login, persist user to DB
user_info = st.session_state.get("user_info", {})
if user_info and user_info.get("email"):
    upsert_user(
        uid          = user_info.get("id", user_info["email"]),
        email        = user_info["email"],
        display_name = user_info.get("name", ""),
        photo_url    = user_info.get("picture", ""),
    )

# ─────────────────────────────────────────────
# AUTHENTICATED APP
# ─────────────────────────────────────────────
user  = get_user()
uid   = user["uid"]
prefs = get_prefs(uid)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    # User pill
    if user["photo_url"]:
        st.markdown(
            f'<div class="user-pill">'
            f'<img class="pill-avatar" src="{user["photo_url"]}">'
            f'{user["display_name"] or user["email"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"👤 **{user['display_name'] or user['email']}**")

    st.markdown("---")

    # Navigation
    page = st.radio("Navigation", ["📰 Feed", "📊 Dashboard", "⚙️ Preferences"], label_visibility="collapsed")

    st.markdown("---")

    # API key
    try:
        api_key = st.secrets.get("NEWS_API_KEY", "")
    except (FileNotFoundError, KeyError):
        api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        api_key = st.text_input("NewsAPI key", type="password")

    if st.button("🔄 Fetch latest news", disabled=not api_key, use_container_width=True):
        with st.spinner("Fetching & processing articles…"):
            try:
                total, added = fetch_and_store(api_key)
                st.success(f"Fetched {total} — {added} new")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    if st.button("Sign out", use_container_width=True):
        logout()

# ─────────────────────────────────────────────
# PAGE: FEED
# ─────────────────────────────────────────────
if page == "📰 Feed":
    st.markdown('<div class="section-header">Your News Feed</div>', unsafe_allow_html=True)

    conn = get_connection()
    c    = conn.cursor()

    # Filter controls
    fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
    with fcol1:
        selected_topic = st.selectbox("Topic", ["All"] + prefs["preferred_topics"])
    with fcol2:
        selected_sentiment = st.multiselect(
            "Sentiment", ALL_SENTIMENTS, default=prefs["preferred_sentiments"]
        )
    with fcol3:
        search = st.text_input("🔍 Search headlines", placeholder="e.g. Fed, Apple, election…")

    if not selected_sentiment:
        selected_sentiment = ALL_SENTIMENTS

    # Build query
    base  = "SELECT id, title, summary, sentiment, publishedAt, source, topic FROM articles WHERE 1=1"
    params = []
    if selected_topic != "All":
        base += " AND topic=?"; params.append(selected_topic)
    if selected_sentiment:
        base += f" AND sentiment IN ({','.join('?'*len(selected_sentiment))})"; params.extend(selected_sentiment)
    if search:
        base += " AND (title LIKE ? OR summary LIKE ?)"; params.extend([f"%{search}%", f"%{search}%"])
    base += " ORDER BY publishedAt DESC LIMIT 50"

    c.execute(base, params)
    rows = c.fetchall()
    conn.close()

    read_ids = get_read_ids(uid)

    if not rows:
        st.info("No articles match your filters. Try fetching the latest news.")
    else:
        # Stats bar
        total_shown = len(rows)
        unread      = sum(1 for r in rows if r[0] not in read_ids)
        s1, s2, s3 = st.columns(3)
        s1.metric("Articles shown", total_shown)
        s2.metric("Unread", unread)
        s3.metric("Sources", len({r[5] for r in rows}))

        st.markdown("<br>", unsafe_allow_html=True)

        for row in rows:
            art_id, title, summary, sentiment, published_at, source, topic = row
            is_read   = art_id in read_ids
            sent_class = {"POSITIVE": "tag-pos", "NEGATIVE": "tag-neg"}.get(sentiment, "tag-neu")
            icon       = SENTIMENT_ICON.get(sentiment, "⚪")
            opacity    = "0.55" if is_read else "1"

            read_label  = "✓ Read" if is_read else f"{icon} {title}"
            title_label = f"{'~~' if is_read else ''}{title}{'~~' if is_read else ''}"
            # Show dimmed title for read articles
            display_title = f"✓ {title}" if is_read else f"{icon} {title}"
            with st.expander(display_title, expanded=False):
                if is_read:
                    st.caption("_You have already read this article._")
                st.markdown(
                    f'<span class="tag {sent_class}">{sentiment}</span> '
                    f'&nbsp;<span style="font-size:12px;color:#94a3b8">{topic} · {source} · {published_at[:10] if published_at else ""}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                st.write(summary or "_No summary available._")

                if not is_read:
                    if st.button("✓ Mark as read", key=f"read_{art_id}"):
                        mark_read(uid, art_id)
                        st.rerun()

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.markdown('<div class="section-header">Intelligence Dashboard</div>', unsafe_allow_html=True)

    # ── Top metrics row ───────────────────────
    stats     = db_stats()
    time_data = reading_time_saved()
    user_stats = get_user_stats(uid)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total articles", stats["total"])
    m2.metric("News sources",   stats["sources"])
    m3.metric("Topics covered", stats["topics"])
    m4.metric("Minutes saved",  f"{time_data['minutes_saved']}m")
    m5.metric("Compression",    f"{time_data['compression_pct']}%")

    st.markdown("---")

    # ── Sentiment trend ───────────────────────
    st.markdown("#### Sentiment Trend (last 7 days)")
    trend_topic = st.selectbox("Filter by topic", ["All"] + ALL_TOPICS, key="trend_topic")
    trend_df    = sentiment_trend(days=7, topic=trend_topic)

    if not trend_df.empty and "date" in trend_df.columns:
        st.line_chart(trend_df.set_index("date")[["POSITIVE", "NEGATIVE", "NEUTRAL"]])
    else:
        st.info("Not enough data yet — fetch more articles across multiple days.")

    st.markdown("---")

    # ── Topic sentiment scores ─────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Topic Sentiment Scores")
        scores_df = topic_sentiment_scores()
        if not scores_df.empty:
            display = scores_df[["topic", "positive_pct", "negative_pct", "net_score"]].copy()
            display.columns = ["Topic", "Positive %", "Negative %", "Net Score"]
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("Fetch articles to see scores.")

    with col_b:
        st.markdown("#### Top Sources")
        src_topic = st.selectbox("Filter", ["All"] + ALL_TOPICS, key="src_topic")
        src_df    = source_breakdown(topic=src_topic, limit=8)
        if not src_df.empty:
            st.bar_chart(src_df.set_index("source")["count"])
        else:
            st.info("No source data yet.")

    st.markdown("---")

    # ── Company mention tracker ────────────────
    st.markdown("#### Company Mention Tracker")
    st.caption("Track how often specific companies appear in the news and their sentiment")

    tracked = prefs.get("tracked_companies", [])
    company_input = st.text_input(
        "Add companies to track (comma-separated)",
        value=", ".join(tracked),
        placeholder="Apple, Tesla, Microsoft, Fed"
    )
    companies = [c.strip() for c in company_input.split(",") if c.strip()]

    if companies:
        mentions_df = company_mentions(companies)
        if not mentions_df.empty and mentions_df["mentions"].sum() > 0:
            # Sentiment score colour coding
            def score_colour(val):
                if val > 0.1:   return "background-color: #dcfce7"
                if val < -0.1:  return "background-color: #fee2e2"
                return "background-color: #dbeafe"

            st.dataframe(
                mentions_df.style.applymap(score_colour, subset=["sentiment_score"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("Sentiment score: +1 = fully positive, -1 = fully negative, 0 = neutral")
        else:
            st.info("None of these companies were found in the current articles. Fetch more news first.")

    st.markdown("---")

    # ── Your personal stats ────────────────────
    st.markdown("#### Your Reading Stats")
    if user_stats["total_read"] == 0:
        st.info("Start reading articles to see your personal stats.")
    else:
        p1, p2, p3 = st.columns(3)
        p1.metric("Articles read",  user_stats["total_read"])
        p2.metric("Time saved",     f"{user_stats['time_saved_min']} min")
        p3.metric("Avg compression", f"{user_stats['avg_compression']}%")

        if user_stats["by_topic"]:
            st.markdown("**Reading breakdown by topic**")
            topic_df = pd.DataFrame(
                list(user_stats["by_topic"].items()),
                columns=["Topic", "Articles read"]
            )
            st.bar_chart(topic_df.set_index("Topic"))

# ─────────────────────────────────────────────
# PAGE: PREFERENCES
# ─────────────────────────────────────────────
elif page == "⚙️ Preferences":
    st.markdown('<div class="section-header">Your Preferences</div>', unsafe_allow_html=True)
    st.caption("Customise your feed. Changes are saved immediately.")

    pref_topics = st.multiselect(
        "Topics you want to follow",
        ALL_TOPICS,
        default=prefs["preferred_topics"],
    )

    pref_sentiments = st.multiselect(
        "Sentiment filter (default for your feed)",
        ALL_SENTIMENTS,
        default=prefs["preferred_sentiments"],
    )

    tracked_raw = st.text_input(
        "Companies to track in the dashboard (comma-separated)",
        value=", ".join(prefs.get("tracked_companies", [])),
        placeholder="Apple, Tesla, Google"
    )
    tracked_companies = [c.strip() for c in tracked_raw.split(",") if c.strip()]

    st.markdown("---")
    st.markdown("**Account**")
    st.write(f"Signed in as **{user['email']}**")

    if st.button("💾 Save preferences", type="primary"):
        if not pref_topics:
            st.error("Select at least one topic.")
        elif not pref_sentiments:
            st.error("Select at least one sentiment.")
        else:
            save_prefs(uid, {
                "preferred_topics":     pref_topics,
                "preferred_sentiments": pref_sentiments,
                "tracked_companies":    tracked_companies,
                "email_briefing":       False,
            })
            st.success("Preferences saved!")
            st.rerun()