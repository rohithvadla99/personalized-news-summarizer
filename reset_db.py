"""
reset_db.py  —  drop and recreate the articles table.

Run only when you want to wipe all stored articles and start fresh.
"""
import sqlite3
import os

from config import DB_PATH   # single source of truth for the path


def reset_db():
    confirm = input(
        "WARNING: This will DELETE all articles from the database.\n"
        "Type 'yes' to continue: "
    ).strip().lower()

    if confirm != "yes":
        print("Aborted — database was not changed.")
        return

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("DROP TABLE IF EXISTS articles")

    c.execute('''
        CREATE TABLE articles (
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
    conn.close()
    print("Database reset. Fresh articles table created.")


if __name__ == "__main__":
    reset_db()
