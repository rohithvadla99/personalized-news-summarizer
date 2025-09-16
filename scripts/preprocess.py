import re

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'<.*?>','', text)
    text = re.sub(r'\s+',' ', text)
    return text.strip()
