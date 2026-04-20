"""Build a training dataset that links news text to company price movements."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).with_name("data")
DEFAULT_PRICES_FILE = DATA_DIR / "daily_prices.csv"
LEGACY_PRICES_FILE = Path(__file__).with_name("daily_prices.csv")
DEFAULT_DATASET_FILE = DATA_DIR / "dataset.csv"


def _parse_date(value: object) -> pd.Timestamp:
    """Convert a value into a normalized YYYY-MM-DD pandas timestamp."""
    return pd.to_datetime(value).normalize()


def _resolve_prices_path(prices_path: str | Path | None = None) -> Path:
    """Prefer data/daily_prices.csv, but allow an explicit or legacy path."""
    if prices_path is not None:
        return Path(prices_path)

    if DEFAULT_PRICES_FILE.exists():
        return DEFAULT_PRICES_FILE

    return LEGACY_PRICES_FILE


def load_data(prices_path: str | Path | None = None) -> pd.DataFrame:
    """Load daily_prices.csv into a cleaned pandas DataFrame."""
    path = _resolve_prices_path(prices_path)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")

    prices_df = pd.read_csv(path)
    required_columns = {"date", "company", "price"}

    if not required_columns.issubset(prices_df.columns):
        raise ValueError("daily_prices.csv must contain columns: date, company, price")

    prices_df = prices_df.copy()
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.normalize()
    prices_df["company"] = prices_df["company"].astype(str).str.strip()
    prices_df["price"] = pd.to_numeric(prices_df["price"], errors="coerce")

    prices_df = prices_df.dropna(subset=["date", "company", "price"])
    prices_df = prices_df.sort_values(["company", "date"]).drop_duplicates(
        subset=["company", "date"],
        keep="last",
    )

    if prices_df.empty:
        raise ValueError("daily_prices.csv has no usable rows.")

    return prices_df


def load_news_data(
    news_items: Iterable[dict[str, object]] | None = None,
    news_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load news data from a list of items or a CSV file with date and text."""
    if news_items is None and news_path is None:
        raise ValueError("Provide either news_items or news_path.")

    if news_items is not None:
        news_df = pd.DataFrame(list(news_items))
    else:
        path = Path(news_path)
        if not path.exists():
            raise FileNotFoundError(f"News file not found: {path}")
        news_df = pd.read_csv(path)

    required_columns = {"date", "text"}
    if not required_columns.issubset(news_df.columns):
        raise ValueError("News data must contain columns: date, text")

    news_df = news_df.copy()
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.normalize()
    news_df["text"] = news_df["text"].astype(str).str.strip()

    news_df = news_df.dropna(subset=["date", "text"])
    news_df = news_df[news_df["text"] != ""]
    news_df = news_df.drop_duplicates(subset=["date", "text"])

    if news_df.empty:
        raise ValueError("News data has no usable rows.")

    return news_df


def get_price(
    prices_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    company: str,
) -> float | None:
    """Return the price on the target date, or the previous trading day if needed."""
    target_timestamp = _parse_date(target_date)
    company_prices = prices_df[prices_df["company"] == company]

    if company_prices.empty:
        return None

    eligible_rows = company_prices[company_prices["date"] <= target_timestamp]
    if eligible_rows.empty:
        return None

    return float(eligible_rows.iloc[-1]["price"])


def _get_future_price(
    prices_df: pd.DataFrame,
    target_date: str | pd.Timestamp,
    company: str,
) -> float | None:
    """Return the first available price on or after the target date."""
    target_timestamp = _parse_date(target_date)
    company_prices = prices_df[prices_df["company"] == company]

    if company_prices.empty:
        return None

    eligible_rows = company_prices[company_prices["date"] >= target_timestamp]
    if eligible_rows.empty:
        return None

    return float(eligible_rows.iloc[0]["price"])


def calculate_label(
    baseline_price: float,
    future_price: float,
    threshold: float = 0.02,
) -> str:
    """Convert percentage price change into Up, Down, or Neutral."""
    change = (future_price - baseline_price) / baseline_price

    if change > threshold:
        return "Up"
    if change < -threshold:
        return "Down"
    return "Neutral"


def _prepare_price_history(
    prices_df: pd.DataFrame,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Cache each company's sorted date/price history for fast lookup."""
    history: dict[str, tuple[pd.Series, pd.Series]] = {}

    for company, group in prices_df.groupby("company", sort=True):
        ordered_group = group.sort_values("date").reset_index(drop=True)
        history[company] = (ordered_group["date"], ordered_group["price"])

    return history


def _lookup_latest_price(
    dates: pd.Series,
    prices: pd.Series,
    target_date: pd.Timestamp,
) -> float | None:
    """Return the latest price on or before target_date from a sorted series."""
    index = int(dates.searchsorted(target_date, side="right") - 1)
    if index < 0:
        return None

    return float(prices.iloc[index])


def _lookup_future_price(
    dates: pd.Series,
    prices: pd.Series,
    target_date: pd.Timestamp,
) -> float | None:
    """Return the first price on or after target_date from a sorted series."""
    index = int(dates.searchsorted(target_date, side="left"))
    if index >= len(dates):
        return None

    return float(prices.iloc[index])


def build_dataset(
    news_items: Iterable[dict[str, object]] | None = None,
    news_path: str | Path | None = None,
    prices_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_DATASET_FILE,
    days_ahead: int = 3,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """Create dataset rows: text, company, movement."""
    print("Starting dataset build...")

    prices_df = load_data(prices_path)
    print(f"Prices loaded: {len(prices_df)} rows")

    news_df = load_news_data(news_items=news_items, news_path=news_path)
    print(f"News loaded: {len(news_df)} rows")

    price_history = _prepare_price_history(prices_df)
    print(f"Companies found: {len(price_history)}")

    dataset_rows: list[tuple[str, str, str]] = []
    seen_rows: set[tuple[str, str, str]] = set()

    for i, news_row in enumerate(news_df.itertuples(index=False), start=1):
        if i == 1 or i % 500 == 0:
            print(f"Processed {i} news...")

        news_date = _parse_date(news_row.date)
        news_text = str(news_row.text).strip()
        future_target_date = news_date + timedelta(days=days_ahead)

        for company, (company_dates, company_prices) in price_history.items():
            baseline_price = _lookup_latest_price(company_dates, company_prices, news_date)
            if baseline_price is None:
                continue

            future_price = _lookup_future_price(
                company_dates,
                company_prices,
                future_target_date,
            )
            if future_price is None:
                continue

            movement = calculate_label(baseline_price, future_price, threshold)
            row = (news_text, company, movement)

            if row not in seen_rows:
                seen_rows.add(row)
                dataset_rows.append(row)

    dataset_df = pd.DataFrame(dataset_rows, columns=["text", "company", "movement"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(output, index=False)

    print(f"Done. Saved {len(dataset_df)} rows to {output}")
    return dataset_df


def main() -> None:
    """Build dataset.csv from a news CSV and daily_prices.csv."""
    parser = argparse.ArgumentParser(description="Build ML labels from news and price data.")
    parser.add_argument(
        "--news",
        required=True,
        help="Path to a CSV file with columns: date,text",
    )
    parser.add_argument(
        "--prices",
        default=None,
        help="Path to daily_prices.csv. Defaults to data/daily_prices.csv if present.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DATASET_FILE),
        help="Where to save dataset.csv",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of days after the news date to measure future movement",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Movement threshold. 0.02 means 2%%.",
    )

    args = parser.parse_args()

    dataset_df = build_dataset(
        news_path=args.news,
        prices_path=args.prices,
        output_path=args.output,
        days_ahead=args.days,
        threshold=args.threshold,
    )

    print(f"Saved {len(dataset_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
