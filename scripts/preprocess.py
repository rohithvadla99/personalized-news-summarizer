import re

def clean_text(text: str) -> str:
    """
    Clean raw article text before NLP processing.
    Removes:
      - HTML tags            e.g. <p>, <b>, <a href="...">
      - NewsAPI truncation   e.g. [+1847 chars]
      - Extra whitespace
    """
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)               # strip HTML tags
    text = re.sub(r"\[\+\d+ chars?\]", "", text)    # strip NewsAPI "[+N chars]"
    text = re.sub(r"\s+", " ", text)                # collapse whitespace
    return text.strip()
