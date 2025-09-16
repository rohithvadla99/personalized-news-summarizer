import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "news.db")  # adjust path if needed

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Drop old table if exists
c.execute("DROP TABLE IF EXISTS articles")

# Recreate table with UNIQUE constraint
c.execute('''
    CREATE TABLE articles (
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
print("Database reset and table created with UNIQUE(title, source)")
