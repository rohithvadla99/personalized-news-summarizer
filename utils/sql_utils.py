import sqlite3
from config import DB_PATH

def get_connection():
    return sqlite3.connect(DB_PATH)

def fetch_all_articles():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, content FROM articles")
    rows = c.fetchall()
    conn.close()
    return rows
