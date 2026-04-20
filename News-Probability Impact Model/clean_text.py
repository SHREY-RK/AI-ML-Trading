"""Simple text preprocessing helpers."""

import string


# A small built-in stopword list keeps the module dependency-free.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def clean_text(text: str) -> str:
    """Lowercase text, remove punctuation and stopwords, and return the result."""
    # Convert to lowercase first so stopword checks are consistent.
    text = text.lower()

    # Remove punctuation characters such as commas, periods, and quotes.
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    words = text.split()
    cleaned_words = [word for word in words if word not in STOPWORDS]

    return " ".join(cleaned_words)
