from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    if not text:
        return "NEUTRAL"
    sentiment = sentiment_analyzer(text)[0]
    return sentiment['label']
