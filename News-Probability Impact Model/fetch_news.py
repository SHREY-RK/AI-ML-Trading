"""Utilities for fetching financial news from NewsAPI."""

from __future__ import annotations

import os
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()


NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_QUERY = "business OR market OR stocks OR earnings"


def fetch_news(date: str) -> list[str]:
    """Fetch financial news for one calendar date."""
    # NewsAPI Everything supports date filtering. Use the exact day for both bounds.
    target_date = datetime.strptime(date, "%Y-%m-%d").date().isoformat()

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise RuntimeError("NEWS_API_KEY environment variable is not set.")

    params = {
        "q": NEWS_QUERY,
        "from": target_date,
        "to": target_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": api_key,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=30)
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "ok":
        message = payload.get("message", "NewsAPI returned an unexpected response.")
        raise RuntimeError(message)

    news_items: list[str] = []

    for article in payload.get("articles", []):
        title = (article.get("title") or "").strip()
        description = (article.get("description") or "").strip()

        if title and description:
            news_items.append(f"{title} - {description}")
        elif title:
            news_items.append(title)
        elif description:
            news_items.append(description)

    return news_items
