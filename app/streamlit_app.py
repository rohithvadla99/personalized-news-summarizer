import streamlit as st
import sqlite3
import os
import sys
import requests
import re
import pandas as pd
from transformers import pipeline

_APP_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_APP_DIR, "..")
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, "scripts")
for _p in [_APP_DIR, _ROOT_DIR, _SCRIPTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from auth       import handle_callback, restore_session, is_logged_in, get_user, logout, render_login
from user_prefs import init_user_tables, upsert_user, get_prefs, save_prefs, mark_read, get_read_ids, get_user_stats
from metrics    import (sentiment_trend, source_breakdown, topic_sentiment_scores,
                        reading_time_saved, company_mentions, db_stats,
                        market_correlation, system_metrics, daily_token_volume)
from fetch_news import fetch_and_store as async_fetch_and_store

st.set_page_config(page_title="NewsIQ", page_icon="📰", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "news.db")

ALL_TOPICS     = ["Sports", "Tech", "Politics", "Business"]
ALL_SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
NEWSAPI_CATEGORIES = {"Sports":"sports","Tech":"technology","Politics":"general","Business":"business"}

TOPIC_COLORS = {
    "Sports":   ("#f0fdf4","#166534","#22c55e"),
    "Tech":     ("#eff6ff","#1e40af","#3b82f6"),
    "Politics": ("#fef9c3","#854d0e","#eab308"),
    "Business": ("#fdf4ff","#6b21a8","#a855f7"),
    "Other":    ("#f8fafc","#334155","#94a3b8"),
}
SENTIMENT_COLOR = {
    "POSITIVE": ("#dcfce7","#166534"),
    "NEGATIVE": ("#fee2e2","#991b1b"),
    "NEUTRAL":  ("#dbeafe","#1e40af"),
}
SENTIMENT_ICON = {"POSITIVE":"🟢","NEGATIVE":"🔴","NEUTRAL":"🔵"}

init_user_tables()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html,body,[class*="css"],.stMarkdown,.stText{font-family:'DM Sans',sans-serif!important}
.section-header{font-family:'DM Serif Display',serif!important;font-size:28px;font-weight:400;
    letter-spacing:-.5px;margin-bottom:4px;padding-bottom:12px;border-bottom:1px solid rgba(128,128,128,.2)}
.user-pill{display:inline-flex;align-items:center;gap:8px;border-radius:20px;padding:4px 12px 4px 4px;
    font-size:13px;background:rgba(128,128,128,.1);margin-bottom:8px}
.pill-avatar{width:28px;height:28px;border-radius:50%;object-fit:cover}
.scale-card{border-radius:10px;padding:16px 20px;border:1px solid rgba(128,128,128,.2);
    text-align:center;background:rgba(128,128,128,.05)}
.scale-card .sc-value{font-family:'DM Serif Display',serif;font-size:26px;font-weight:400;line-height:1.1}
.scale-card .sc-label{font-size:11px;opacity:.6;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
#MainMenu{visibility:hidden}footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

def clean_text(text):
    if not text: return ""
    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"\[\+\d+ chars?\]","",text)
    return re.sub(r"\s+"," ",text).strip()

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute('''CREATE TABLE IF NOT EXISTS articles(
        id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT,content TEXT,description TEXT,
        source TEXT,publishedAt TEXT,summary TEXT,sentiment TEXT,topic TEXT,
        url TEXT,tokens_processed INTEGER DEFAULT 0,UNIQUE(title,source))''')
    conn.execute('''CREATE TABLE IF NOT EXISTS fetch_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT,fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
        articles_fetched INTEGER DEFAULT 0,articles_new INTEGER DEFAULT 0,
        tokens_processed INTEGER DEFAULT 0,sources_hit INTEGER DEFAULT 0,
        avg_compression REAL DEFAULT 0,duration_sec REAL DEFAULT 0)''')
    conn.commit()
    for col,defn in [("url","TEXT"),("tokens_processed","INTEGER DEFAULT 0")]:
        try: conn.execute(f"ALTER TABLE articles ADD COLUMN {col} {defn}"); conn.commit()
        except: pass
    try: conn.execute("ALTER TABLE fetch_log ADD COLUMN duration_sec REAL DEFAULT 0"); conn.commit()
    except: pass
    return conn

# ── Login gate ────────────────────────────────
handle_callback()
restore_session()

if not is_logged_in():
    render_login()
    st.stop()

_u = get_user()
if _u.get("email"):
    upsert_user(uid=_u["uid"], email=_u["email"],
                display_name=_u["display_name"], photo_url=_u["photo_url"])

user  = get_user()
uid   = user["uid"]
prefs = get_prefs(uid)

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    if user["photo_url"]:
        st.markdown(f'<div class="user-pill"><img class="pill-avatar" src="{user["photo_url"]}">'
                    f'<span>{user["display_name"] or user["email"]}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown(f"👤 **{user['display_name'] or user['email']}**")

    page = st.radio("Navigation", ["📰 Feed","📊 Dashboard","⚙️ Preferences"], label_visibility="collapsed")
    st.markdown("---")

    try:    api_key = st.secrets.get("NEWS_API_KEY","")
    except: api_key = os.environ.get("NEWS_API_KEY","")
    if not api_key:
        api_key = st.text_input("NewsAPI key", type="password")

    if st.button("🔄 Fetch latest news", disabled=not api_key, use_container_width=True):
        with st.spinner("Fetching & scraping concurrently…"):
            try:
                total, added, duration = async_fetch_and_store(api_key)
                st.success(f"✓ {added} new in {duration}s")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    if st.button("Sign out", use_container_width=True):
        logout()

# ═══════════════════════════════════════════
# FEED
# ═══════════════════════════════════════════
if page == "📰 Feed":
    st.markdown('<div class="section-header">Your News Feed</div>', unsafe_allow_html=True)

    conn = get_connection(); c = conn.cursor()
    f1,f2,f3 = st.columns([1,1,2])
    with f1: selected_topic = st.selectbox("Topic", ["All"]+prefs["preferred_topics"])
    with f2: selected_sentiment = st.multiselect("Sentiment", ALL_SENTIMENTS, default=prefs["preferred_sentiments"])
    with f3: search = st.text_input("🔍 Search headlines", placeholder="e.g. Fed, Apple, election…")

    if not selected_sentiment: selected_sentiment = ALL_SENTIMENTS

    base = "SELECT id,title,summary,sentiment,publishedAt,source,topic,url FROM articles WHERE 1=1"
    params = []
    if selected_topic != "All":      base += " AND topic=?";      params.append(selected_topic)
    if selected_sentiment:           base += f" AND sentiment IN ({','.join('?'*len(selected_sentiment))})"; params.extend(selected_sentiment)
    if search:                       base += " AND (title LIKE ? OR summary LIKE ?)"; params.extend([f"%{search}%"]*2)
    base += " ORDER BY publishedAt DESC LIMIT 50"

    c.execute(base, params); rows = c.fetchall(); conn.close()
    read_ids = get_read_ids(uid)

    if not rows:
        st.info("No articles yet — hit **Fetch latest news** in the sidebar.")
    else:
        total_shown = len(rows)
        unread      = sum(1 for r in rows if r[0] not in read_ids)
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Shown",  total_shown)
        s2.metric("Unread", unread)
        s3.metric("Read",   total_shown - unread)
        s4.metric("Sources", len({r[5] for r in rows if r[5]}))

        st.markdown("<br>", unsafe_allow_html=True)

        # ── HTML cards ────────────────────────
        cards_html = []
        for row in rows:
            art_id,title,summary,sentiment,published_at,source,topic,url = row
            is_read  = art_id in read_ids
            sbg,sc   = SENTIMENT_COLOR.get(sentiment or "NEUTRAL",("#dbeafe","#1e40af"))
            tbg,tc,_ = TOPIC_COLORS.get(topic or "Other", TOPIC_COLORS["Other"])
            date_str = (published_at or "")[:10]
            opacity  = "0.45" if is_read else "1"
            read_tag = '<span style="font-size:11px;color:#94a3b8;margin-left:6px">✓ read</span>' if is_read else ""
            link_btn = (f'<a href="{url}" target="_blank" style="display:inline-flex;align-items:center;'
                        f'gap:5px;background:#0f172a;color:#fff;padding:6px 14px;border-radius:6px;'
                        f'font-size:12px;font-weight:500;text-decoration:none;margin-top:10px">'
                        f'Read full article →</a>') if url else ""

            cards_html.append(f"""
<div style="border:1px solid rgba(128,128,128,.18);border-radius:12px;padding:20px 24px;
            margin-bottom:10px;opacity:{opacity};background:rgba(128,128,128,.03)">
  <p style="font-size:15px;font-weight:600;margin:0 0 8px;line-height:1.4">{title}{read_tag}</p>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px">
    <span style="display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;
        font-weight:600;background:{sbg};color:{sc}">{sentiment or "NEUTRAL"}</span>
    <span style="display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;
        font-weight:600;background:{tbg};color:{tc}">{topic or "Other"}</span>
    <span style="font-size:12px;opacity:.5">{source} · {date_str}</span>
  </div>
  <p style="font-size:14px;line-height:1.65;margin:0 0 4px;opacity:.85">{summary or "<em>No summary available.</em>"}</p>
  {link_btn}
</div>""")

        st.markdown("".join(cards_html), unsafe_allow_html=True)

        # ── Mark as read buttons ──────────────
        unread_rows = [(r[0], r[1]) for r in rows if r[0] not in read_ids]
        if unread_rows:
            st.markdown("<br>**Mark as read:**", unsafe_allow_html=True)
            cols = st.columns(min(3, len(unread_rows)))
            for i,(aid,btitle) in enumerate(unread_rows[:9]):
                with cols[i%3]:
                    short = btitle[:38]+"…" if len(btitle)>38 else btitle
                    if st.button(f"✓ {short}", key=f"read_{aid}", use_container_width=True):
                        mark_read(uid, aid); st.rerun()

# ═══════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown('<div class="section-header">Intelligence Dashboard</div>', unsafe_allow_html=True)

    stats      = db_stats()
    time_data  = reading_time_saved()
    user_stats = get_user_stats(uid)
    sys_m      = system_metrics()

    st.markdown("##### Platform Overview")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total articles",  stats["total"])
    m2.metric("News sources",    stats["sources"])
    m3.metric("Topics",          stats["topics"])
    m4.metric("Minutes saved",   f"{time_data['minutes_saved']}m")
    m5.metric("Avg compression", f"{time_data['compression_pct']}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### System Scale")

    def scale_card(v,l):
        return f'<div class="scale-card"><div class="sc-value">{v}</div><div class="sc-label">{l}</div></div>'

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5,c6],[
        (f"{sys_m['total_tokens']:,}","Tokens processed"),
        (f"{sys_m['total_words']:,}","Words processed"),
        (str(sys_m["unique_sources"]),"Sources monitored"),
        (str(sys_m["total_users"]),"Registered users"),
        (str(sys_m["total_reads"]),"Articles read"),
        (str(sys_m["total_runs"]),"Fetch runs"),
    ]):
        col.markdown(scale_card(val,lbl), unsafe_allow_html=True)

    vol_df = daily_token_volume(days=30)
    if not vol_df.empty and len(vol_df) > 1:
        st.markdown("<br>**Daily token throughput (last 30 days)**", unsafe_allow_html=True)
        st.bar_chart(vol_df.set_index("date")["tokens"])

    st.markdown("---")
    st.markdown("##### Sentiment Trend")
    t1,t2 = st.columns([1,3])
    with t1:
        trend_days  = st.selectbox("Window",[7,14,30],key="trend_days")
        trend_topic = st.selectbox("Topic",["All"]+ALL_TOPICS,key="trend_topic")
    with t2:
        trend_df = sentiment_trend(days=trend_days, topic=trend_topic)
        if not trend_df.empty and "date" in trend_df.columns:
            st.line_chart(trend_df.set_index("date")[["POSITIVE","NEGATIVE","NEUTRAL"]],
                         color=["#22c55e","#ef4444","#3b82f6"])
        else:
            st.info("Not enough data yet — fetch news across multiple days.")

    st.markdown("---")
    col_a,col_b = st.columns(2)
    with col_a:
        st.markdown("##### Topic Sentiment Scores")
        scores_df = topic_sentiment_scores()
        if not scores_df.empty:
            display = scores_df[["topic","positive_pct","negative_pct","net_score"]].copy()
            display.columns = ["Topic","Positive %","Negative %","Net Score"]
            def color_net(val):
                try:
                    v = float(val)
                    if v>0: return "color:#166534;font-weight:600"
                    if v<0: return "color:#991b1b;font-weight:600"
                except: pass
                return ""
            st.dataframe(display.style.applymap(color_net,subset=["Net Score"]),
                        use_container_width=True, hide_index=True)
        else:
            st.info("Fetch articles to see scores.")

    with col_b:
        st.markdown("##### Top Sources")
        src_topic = st.selectbox("Filter by topic",["All"]+ALL_TOPICS,key="src_topic")
        src_df    = source_breakdown(topic=src_topic, limit=8)
        if not src_df.empty: st.bar_chart(src_df.set_index("source")["count"])
        else: st.info("No source data yet.")

    st.markdown("---")
    st.markdown("##### 📈 Sentiment vs Market Correlation")
    st.caption("News sentiment vs S&P 500 / Tech / Finance daily % change")
    market_days = st.slider("Lookback (days)",7,90,30,7,key="market_days")
    with st.spinner("Fetching market data…"):
        mdata = market_correlation(days=market_days)

    if mdata.get("error"):
        st.info(mdata["error"])
    else:
        df_m,corrs,tickers = mdata["combined_df"],mdata["correlations"],mdata["tickers"]
        cc = st.columns(len(corrs))
        for i,(label,val) in enumerate(corrs.items()):
            strength = "Strong" if abs(val)>.5 else "Moderate" if abs(val)>.3 else "Weak"
            cc[i].metric(label, f"{val:+.3f}", f"{'↑' if val>0 else '↓'} {strength}")
        tabs = st.tabs(list(corrs.keys()))
        for tab,ticker_label in zip(tabs,tickers):
            with tab:
                pct_col = f"{ticker_label} %"
                if pct_col not in df_m.columns: st.info(f"No data for {ticker_label}"); continue
                chart_df = df_m[["date","sentiment_score",pct_col]].dropna().copy()
                chart_df = chart_df.rename(columns={"sentiment_score":"News Sentiment",pct_col:f"{ticker_label} Daily %"}).set_index("date")
                for col in chart_df.columns:
                    mn,mx = chart_df[col].min(),chart_df[col].max()
                    if mx!=mn: chart_df[f"{col} (norm)"]=(chart_df[col]-mn)/(mx-mn)
                norm_cols=[c for c in chart_df.columns if "norm" in c]
                if norm_cols:
                    st.line_chart(chart_df[norm_cols])
                    st.caption(f"Normalised 0–1. Pearson r = **{corrs.get(ticker_label,'N/A')}**")

    st.markdown("---")
    st.markdown("##### Company Mention Tracker")
    tracked = prefs.get("tracked_companies",[])
    company_input = st.text_input("Companies (comma-separated)",value=", ".join(tracked),placeholder="Apple, Tesla, Microsoft")
    companies = [c.strip() for c in company_input.split(",") if c.strip()]
    if companies:
        mentions_df = company_mentions(companies)
        if not mentions_df.empty and mentions_df["mentions"].sum()>0:
            def sc(val):
                if val>.1:  return "background-color:#dcfce7;color:#166534"
                if val<-.1: return "background-color:#fee2e2;color:#991b1b"
                return "background-color:#dbeafe;color:#1e40af"
            st.dataframe(mentions_df.style.applymap(sc,subset=["sentiment_score"]),
                        use_container_width=True, hide_index=True)
            st.caption("Score: +1=positive, −1=negative, 0=neutral")
        else:
            st.info("None of these companies found. Fetch more news first.")

    st.markdown("---")
    st.markdown("##### Your Reading Stats")
    if user_stats["total_read"]==0:
        st.info("Mark articles as read in the Feed to see your stats.")
    else:
        p1,p2,p3 = st.columns(3)
        p1.metric("Articles read",   user_stats["total_read"])
        p2.metric("Time saved",      f"{user_stats['time_saved_min']} min")
        p3.metric("Avg compression", f"{user_stats['avg_compression']}%")
        if user_stats["by_topic"]:
            st.bar_chart(pd.DataFrame(list(user_stats["by_topic"].items()),columns=["Topic","Articles read"]).set_index("Topic"))

# ═══════════════════════════════════════════
# PREFERENCES
# ═══════════════════════════════════════════
elif page == "⚙️ Preferences":
    st.markdown('<div class="section-header">Preferences</div>', unsafe_allow_html=True)
    st.caption("Changes are saved when you click Save.")

    pref_topics = st.multiselect("Topics to follow", ALL_TOPICS, default=prefs["preferred_topics"])
    pref_sentiments = st.multiselect("Default sentiment filter", ALL_SENTIMENTS, default=prefs["preferred_sentiments"])
    tracked_raw = st.text_input("Companies to track (comma-separated)",
                                value=", ".join(prefs.get("tracked_companies",[])),
                                placeholder="Apple, Tesla, Google")
    tracked_companies = [c.strip() for c in tracked_raw.split(",") if c.strip()]

    st.markdown("---")
    st.markdown("**Account**")
    st.write(f"Signed in as **{user['email']}**")

    if st.button("💾 Save preferences", type="primary"):
        if not pref_topics:     st.error("Select at least one topic.")
        elif not pref_sentiments: st.error("Select at least one sentiment.")
        else:
            save_prefs(uid,{"preferred_topics":pref_topics,"preferred_sentiments":pref_sentiments,
                            "tracked_companies":tracked_companies,"email_briefing":False})
            st.success("Preferences saved!")
            st.rerun()
