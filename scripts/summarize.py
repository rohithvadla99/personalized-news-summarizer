from transformers import pipeline

# Loaded once at module import; shared across all callers.
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text: str) -> str:
    """
    Summarize text using BART-large-CNN.

    Fixes vs original:
    - Guards against text that is too short for the model (was crashing with
      ValueError when min_length > number of tokens in input).
    - Dynamically sets min_length so it is always less than the input length.
    - Truncates very long inputs to avoid exceeding the model's token limit.
    """
    if not text:
        return ""

    words = text.split()

    # Too short to summarize meaningfully — return as-is
    if len(words) < 40:
        return text

    # Truncate to 600 words to stay within BART's 1024-token limit
    if len(words) > 600:
        text = " ".join(words[:600])
        words = words[:600]

    # min_length must always be less than max_length and less than input length
    min_len = min(30, max(1, len(words) // 4))

    try:
        result = summarizer(text, max_length=120, min_length=min_len, do_sample=False)
        return result[0]["summary_text"]
    except Exception:
        # Graceful fallback: return the first 300 characters
        return text[:300]
