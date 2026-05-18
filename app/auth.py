"""
auth.py — Google OAuth with DB-backed session tokens.

Session persistence works by:
1. On login: generate a random token, store it in the DB with user info,
   and append ?session=TOKEN to the URL
2. On every reload: read ?session= from the URL, look up user in DB,
   restore session_state
3. On logout: delete the token from DB and clear the URL param

This avoids all cookie libraries and works reliably in Streamlit.
"""

import os
import json
import secrets
import sqlite3
import requests
import streamlit as st
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import DB_PATH


# ── DB setup ───────────────────────────────────────────────────────────────
def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token        TEXT PRIMARY KEY,
            uid          TEXT,
            email        TEXT,
            display_name TEXT,
            photo_url    TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn


def _save_token(token: str, user: dict) -> None:
    conn = _get_conn()
    conn.execute('''
        INSERT OR REPLACE INTO sessions (token, uid, email, display_name, photo_url)
        VALUES (?, ?, ?, ?, ?)
    ''', (token, user["uid"], user["email"], user["display_name"], user["photo_url"]))
    conn.commit()
    conn.close()


def _load_token(token: str) -> dict | None:
    conn = _get_conn()
    c = conn.cursor()
    c.execute('SELECT uid, email, display_name, photo_url FROM sessions WHERE token=?', (token,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"uid": row[0], "email": row[1], "display_name": row[2], "photo_url": row[3]}
    return None


def _delete_token(token: str) -> None:
    conn = _get_conn()
    conn.execute('DELETE FROM sessions WHERE token=?', (token,))
    conn.commit()
    conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────
def _redirect_uri() -> str:
    try:
        uri = st.secrets.get("REDIRECT_URI", "")
        if uri:
            return uri
    except Exception:
        pass
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    return f"http://localhost:{port}"


def _get_creds() -> tuple[str, str]:
    try:
        return st.secrets["GOOGLE_CLIENT_ID"], st.secrets["GOOGLE_CLIENT_SECRET"]
    except Exception:
        return os.environ.get("GOOGLE_CLIENT_ID", ""), os.environ.get("GOOGLE_CLIENT_SECRET", "")


def _set_session(user: dict) -> None:
    st.session_state["user_email"]        = user["email"]
    st.session_state["user_display_name"] = user["display_name"]
    st.session_state["user_photo_url"]    = user["photo_url"]
    st.session_state["user_uid"]          = user["uid"]


# ── Core auth functions ────────────────────────────────────────────────────
def restore_session() -> bool:
    """
    On every page load: check if session_state is populated,
    or if ?session= token exists in the URL and is valid in the DB.
    """
    if st.session_state.get("user_email"):
        return True

    token = st.query_params.get("session", "")
    if not token:
        return False

    user = _load_token(token)
    if user:
        _set_session(user)
        st.session_state["session_token"] = token
        return True

    # Token not found — clear it from URL
    st.query_params.clear()
    return False


def handle_callback() -> bool:
    """
    If ?code= is in the URL (Google redirect), exchange it for user info,
    create a session token, and redirect to /?session=TOKEN.
    Skips if a ?session= token is already present (page reload case).
    """
    # Already have a session token — don't try to reuse the OAuth code
    if st.query_params.get("session", ""):
        return False

    auth_code = st.query_params.get("code", "")
    if not auth_code:
        return False

    client_id, client_secret = _get_creds()

    # Exchange code for access token
    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code":          auth_code,
            "client_id":     client_id,
            "client_secret": client_secret,
            "redirect_uri":  _redirect_uri(),
            "grant_type":    "authorization_code",
        },
        timeout=10,
    )

    if not token_resp.ok:
        st.error(f"Google sign-in failed: {token_resp.text}")
        return False

    access_token = token_resp.json().get("access_token", "")

    # Get user profile
    user_resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )

    if not user_resp.ok:
        st.error("Could not fetch your Google profile.")
        return False

    info = user_resp.json()
    user = {
        "email":        info.get("email", ""),
        "display_name": info.get("name", ""),
        "photo_url":    info.get("picture", ""),
        "uid":          info.get("id", info.get("email", "")),
    }

    # Create session token and save to DB
    token = secrets.token_urlsafe(32)
    _save_token(token, user)
    _set_session(user)
    st.session_state["session_token"] = token

    # Redirect to app with session token in URL
    st.query_params["session"] = token
    return True


def get_login_url() -> str:
    client_id, _ = _get_creds()
    state = secrets.token_urlsafe(16)
    st.session_state["oauth_state"] = state
    params = "&".join([
        f"client_id={client_id}",
        f"redirect_uri={_redirect_uri()}",
        "response_type=code",
        "scope=openid%20email%20profile",
        f"state={state}",
        "access_type=offline",
        "prompt=select_account",
    ])
    return f"https://accounts.google.com/o/oauth2/v2/auth?{params}"


def is_logged_in() -> bool:
    return bool(st.session_state.get("user_email"))


def get_user() -> dict:
    return {
        "email":        st.session_state.get("user_email", ""),
        "display_name": st.session_state.get("user_display_name", ""),
        "photo_url":    st.session_state.get("user_photo_url", ""),
        "uid":          st.session_state.get("user_uid", ""),
    }


def logout() -> None:
    token = st.session_state.get("session_token", "")
    if token:
        _delete_token(token)
    for key in ["user_email", "user_display_name", "user_photo_url",
                "user_uid", "session_token", "oauth_state"]:
        st.session_state.pop(key, None)
    st.query_params.clear()
    st.rerun()


def render_login() -> None:
    login_url = get_login_url()
    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;
                gap:20px;padding:60px 0;font-family:'DM Sans',sans-serif">
        <div style="font-size:52px;line-height:1">📰</div>
        <div style="font-size:28px;font-weight:600;color:#0f172a;
                    font-family:Georgia,serif;letter-spacing:-0.5px">NewsIQ</div>
        <div style="font-size:14px;color:#64748b;text-align:center;max-width:280px">
            Your personalised news intelligence platform
        </div>
        <a href="{login_url}" target="_self" style="
            display:inline-flex;align-items:center;gap:12px;
            background:#fff;border:1.5px solid #dadce0;border-radius:8px;
            padding:12px 28px;font-size:15px;font-weight:500;color:#3c4043;
            text-decoration:none;box-shadow:0 1px 3px rgba(0,0,0,.1);
            margin-top:8px">
            <svg width="20" height="20" viewBox="0 0 24 24">
                <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
                <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
                <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
        </a>
    </div>
    """, unsafe_allow_html=True)