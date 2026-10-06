import re


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()
