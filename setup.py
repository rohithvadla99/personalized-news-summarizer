import os
import sqlite3

# ----- Config -----
DB_FOLDER = "data"
DB_FILE = "news.db"
DB_PATH = os.path.join(DB_FOLDER, DB_FILE)

# ----- Create folder -----
os.makedirs(DB_FOLDER, exist_ok=True)
print(f"Folder '{DB_FOLDER}' ensured.")

# ----- Connect to SQLite -----
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
print(f"Database '{DB_PATH}' connected/created.")

# ----- Create table -----
c.execute('''
CREATE TABLE IF NOT EXISTS articles(
    id INTEGER PRIMARY KEY,
    title TEXT,
    content TEXT,
    description TEXT,
    source TEXT,
    publishedAt TEXT,
    summary TEXT,
    sentiment TEXT
)
''')
conn.commit()
conn.close()
print("Table 'articles' created (if it did not exist).")

print("Setup complete. You can now run fetch_news.py and update_db.py.")
