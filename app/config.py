import os
from dotenv import load_dotenv
load_dotenv()

NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "data", "news.db")

# ── Topics & keyword classifier ────────────────────────────────────────────
TOPICS = ["Sports", "Tech", "Politics", "Business"]

TOPIC_KEYWORDS = {
    "Tech":     ["AI", "software", "Apple", "Google", "chip", "tech", "startup", "cyber", "robot"],
    "Sports":   ["NBA", "NFL", "match", "tournament", "score", "athlete", "league", "cup", "FIFA"],
    "Politics": ["president", "congress", "election", "senate", "democrat", "republican", "bill", "vote"],
    "Business": ["earnings", "market", "stock", "IPO", "revenue", "economy", "trade", "fed", "inflation"],
}
