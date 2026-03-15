from transformers import pipeline

# Loaded once at module import; shared across all callers.
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text: str) -> str:
    """
    Return sentiment label for the given text.

    Fixes vs original:
    - Truncates input to 400 words before calling the model.
      DistilBERT has a hard 512-token limit; passing a full article
      would silently truncate or crash depending on the model version.
    - Always returns UPPER-CASE labels ("POSITIVE", "NEGATIVE", "NEUTRAL")
      so DB values are consistent. The original returned mixed-case
      ("Neutral" fallback vs "POSITIVE" from the model).
    """
    if not text:
        return "NEUTRAL"

    # Truncate to keep well under the 512-token limit
    truncated = " ".join(text.split()[:400])

    try:
        label = sentiment_analyzer(truncated)[0]["label"]
        return label.upper()
    except Exception:
        return "NEUTRAL"
