from asset_model import predict as predict_asset_probabilities
from asset_model import load_model as load_asset_model
from clean_text import clean_text
from fetch_news import fetch_news


def _get_top_movement(probabilities):
    return max(probabilities.items(), key=lambda item: item[1])


def run_pipeline(date: str):
    load_asset_model()

    news_items = fetch_news(date)

    if not news_items:
        print(f"No news found for {date}.")
        return

    for news_text in news_items:
        cleaned_text = clean_text(news_text)
        if not cleaned_text:
            continue

        print(f'News: "{news_text}"\n')

        company_probabilities = predict_asset_probabilities(cleaned_text)

        for company, probabilities in company_probabilities.items():
            movement, probability = _get_top_movement(probabilities)
            print(f"{company} -> {movement} {probability:.2f}")

        print()


def main():
    date = input("Enter date (YYYY-MM-DD): ").strip()

    try:
        run_pipeline(date)
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()