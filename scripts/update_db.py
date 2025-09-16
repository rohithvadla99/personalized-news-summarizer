from utils.sql_utils import get_connection
from scripts.preprocess import clean_text
from scripts.summarize import summarize
from scripts.sentiment import analyze_sentiment

def update_articles():
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT id, content FROM articles')
    rows = c.fetchall()
    for article_id, content in rows:
        cleaned = clean_text(content)
        summary = summarize(cleaned)
        sentiment = analyze_sentiment(cleaned)
        c.execute('UPDATE articles SET summary=?, sentiment=? WHERE id=?', (summary, sentiment, article_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    update_articles()
