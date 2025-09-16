import sqlite3

DB_PATH = "news.db"

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
            topic TEXT
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized with 'articles' table.")
