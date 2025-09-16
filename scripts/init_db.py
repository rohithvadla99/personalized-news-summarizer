import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "news.db")

def init_db():
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
            topic TEXT,
            UNIQUE(title, source)
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized with UNIQUE constraint on (title, source).")
