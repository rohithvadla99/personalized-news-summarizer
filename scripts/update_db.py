import sqlite3

from config     import DB_PATH
from preprocess import clean_text
from summarize  import summarize
from sentiment  import analyze_sentiment


def update_articles():
    """
    Re-run the NLP pipeline on every article already in the DB and
    update its summary and sentiment columns.

    Fixes vs original:
    - Removed imports from non-existent packages (utils.sql_utils,
      scripts.preprocess, scripts.summarize, scripts.sentiment).
      Those modules do not exist — every import was a ModuleNotFoundError.
    - Now imports directly from the sibling modules in this project.
    - Uses DB_PATH from config instead of get_connection() phantom import.
    - Skips articles with no content to avoid crashing the summarizer.
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("SELECT id, content FROM articles")
    rows = c.fetchall()

    updated = 0
    for article_id, content in rows:
        if not content:
            continue  # nothing to process

        cleaned   = clean_text(content)
        summary   = summarize(cleaned)
        sentiment = analyze_sentiment(cleaned)

        c.execute(
            "UPDATE articles SET summary=?, sentiment=? WHERE id=?",
            (summary, sentiment, article_id),
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Updated {updated} articles.")


if __name__ == "__main__":
    update_articles()
