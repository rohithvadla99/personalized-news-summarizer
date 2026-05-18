"""
user_prefs.py — per-user preferences stored in the same SQLite DB.

Schema:
  users          — uid, email, display_name, created_at
  user_prefs     — uid, preferred_topics (JSON), preferred_sentiments (JSON),
                   tracked_companies (JSON), email_briefing (bool)
  read_history   — uid, article_id, read_at
"""

import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_user_tables() -> None:
    """Create user-related tables if they don't exist."""
    conn = get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            uid          TEXT PRIMARY KEY,
            email        TEXT UNIQUE,
            display_name TEXT,
            photo_url    TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login   TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS user_prefs (
            uid                  TEXT PRIMARY KEY,
            preferred_topics     TEXT DEFAULT '["Sports","Tech","Politics","Business"]',
            preferred_sentiments TEXT DEFAULT '["POSITIVE","NEGATIVE","NEUTRAL"]',
            tracked_companies    TEXT DEFAULT '[]',
            email_briefing       INTEGER DEFAULT 0,
            FOREIGN KEY (uid) REFERENCES users(uid)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS read_history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            uid        TEXT,
            article_id INTEGER,
            read_at    TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(uid, article_id),
            FOREIGN KEY (uid) REFERENCES users(uid),
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
    ''')

    conn.commit()
    conn.close()


def upsert_user(uid: str, email: str, display_name: str, photo_url: str = "") -> None:
    """Create or update a user record on login."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        INSERT INTO users (uid, email, display_name, photo_url, created_at, last_login)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(uid) DO UPDATE SET
            display_name = excluded.display_name,
            photo_url    = excluded.photo_url,
            last_login   = excluded.last_login
    ''', (uid, email, display_name, photo_url, datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))

    # Create default prefs row if it doesn't exist
    c.execute('''
        INSERT OR IGNORE INTO user_prefs (uid) VALUES (?)
    ''', (uid,))

    conn.commit()
    conn.close()


def get_prefs(uid: str) -> dict:
    """Return user preferences as a dict."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT preferred_topics, preferred_sentiments, tracked_companies, email_briefing FROM user_prefs WHERE uid=?', (uid,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {
            "preferred_topics":     ["Sports", "Tech", "Politics", "Business"],
            "preferred_sentiments": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
            "tracked_companies":    [],
            "email_briefing":       False,
        }
    return {
        "preferred_topics":     json.loads(row[0]),
        "preferred_sentiments": json.loads(row[1]),
        "tracked_companies":    json.loads(row[2]),
        "email_briefing":       bool(row[3]),
    }


def save_prefs(uid: str, prefs: dict) -> None:
    """Persist user preferences."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        INSERT INTO user_prefs (uid, preferred_topics, preferred_sentiments, tracked_companies, email_briefing)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(uid) DO UPDATE SET
            preferred_topics     = excluded.preferred_topics,
            preferred_sentiments = excluded.preferred_sentiments,
            tracked_companies    = excluded.tracked_companies,
            email_briefing       = excluded.email_briefing
    ''', (
        uid,
        json.dumps(prefs["preferred_topics"]),
        json.dumps(prefs["preferred_sentiments"]),
        json.dumps(prefs["tracked_companies"]),
        int(prefs["email_briefing"]),
    ))
    conn.commit()
    conn.close()


def mark_read(uid: str, article_id: int) -> None:
    """Record that a user has read an article."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('''
        INSERT OR IGNORE INTO read_history (uid, article_id) VALUES (?, ?)
    ''', (uid, article_id))
    conn.commit()
    conn.close()


def get_read_ids(uid: str) -> set:
    """Return set of article IDs the user has already read."""
    conn = get_conn()
    c = conn.cursor()
    c.execute('SELECT article_id FROM read_history WHERE uid=?', (uid,))
    ids = {row[0] for row in c.fetchall()}
    conn.close()
    return ids


def get_user_stats(uid: str) -> dict:
    """Return stats for the user's usage dashboard."""
    conn = get_conn()
    c = conn.cursor()

    # Total articles read
    c.execute('SELECT COUNT(*) FROM read_history WHERE uid=?', (uid,))
    total_read = c.fetchone()[0]

    # Articles read per topic
    c.execute('''
        SELECT a.topic, COUNT(*) as cnt
        FROM read_history rh
        JOIN articles a ON a.id = rh.article_id
        WHERE rh.uid = ?
        GROUP BY a.topic
        ORDER BY cnt DESC
    ''', (uid,))
    by_topic = {row[0]: row[1] for row in c.fetchall()}

    # Sentiment breakdown of articles read
    c.execute('''
        SELECT a.sentiment, COUNT(*) as cnt
        FROM read_history rh
        JOIN articles a ON a.id = rh.article_id
        WHERE rh.uid = ?
        GROUP BY a.sentiment
    ''', (uid,))
    by_sentiment = {row[0]: row[1] for row in c.fetchall()}

    # Average compression (words saved)
    c.execute('''
        SELECT AVG(
            CASE WHEN a.content != '' AND a.summary != ''
            THEN (LENGTH(a.content) - LENGTH(a.summary)) * 1.0 / LENGTH(a.content)
            ELSE 0 END
        )
        FROM read_history rh
        JOIN articles a ON a.id = rh.article_id
        WHERE rh.uid = ?
    ''', (uid,))
    avg_compression = c.fetchone()[0] or 0

    conn.close()

    return {
        "total_read":      total_read,
        "by_topic":        by_topic,
        "by_sentiment":    by_sentiment,
        "avg_compression": round(avg_compression * 100, 1),
        "time_saved_min":  round(total_read * 4 * avg_compression),  # avg 4 min/article
    }