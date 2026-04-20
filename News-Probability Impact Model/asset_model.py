"""Asset-wise movement model with dataset-based training and model persistence."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

from clean_text import clean_text

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "asset_model.py requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


DATA_DIR = Path(__file__).with_name("data")
DATASET_FILE = DATA_DIR / "dataset.csv"
MODELS_DIR = Path(__file__).with_name("models")
MODEL_FILE = MODELS_DIR / "asset_model.pkl"
MOVEMENTS = ["Up", "Down", "Neutral"]

# Cached objects for the current Python process.
MODEL: Pipeline | None = None
COMPANIES: list[str] = []


def _make_label(company: str, movement: str) -> str:
    """Join company and movement into one label for the classifier."""
    return f"{company}::{movement}"


def _load_dataset(dataset_path: str | Path = DATASET_FILE) -> list[tuple[str, str, str]]:
    """Load and validate rows from dataset.csv."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Build it first with dataset_builder.py."
        )

    rows: list[tuple[str, str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"text", "company", "movement"}

        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError("Dataset must contain columns: text, company, movement")

        for row in reader:
            raw_text = (row.get("text") or "").strip()
            company = (row.get("company") or "").strip()
            movement = (row.get("movement") or "").strip().title()

            if not raw_text or not company or movement not in MOVEMENTS:
                raise ValueError(
                    "Each row must have non-empty text/company and movement as Up, Down, or Neutral."
                )

            # Train on cleaned text so training and prediction use the same format.
            cleaned_text = clean_text(raw_text)
            if cleaned_text:
                rows.append((cleaned_text, company, movement))

    if not rows:
        raise ValueError("Dataset file has no usable rows.")

    return rows


def train_model(dataset_path: str | Path = DATASET_FILE, save: bool = True) -> Pipeline:
    """Train the model from dataset.csv and optionally save it to disk."""
    global MODEL, COMPANIES

    dataset = _load_dataset(dataset_path)
    texts = [text for text, _company, _movement in dataset]
    labels = [_make_label(company, movement) for _text, company, movement in dataset]

    # Keep the company names so predict() can return one block per company.
    COMPANIES = sorted({company for _text, company, _movement in dataset})

    MODEL = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )

    MODEL.fit(texts, labels)

    if save:
        save_model()

    return MODEL


def save_model(model_path: str | Path = MODEL_FILE) -> Path:
    """Save the trained model and company list to disk."""
    if MODEL is None:
        raise RuntimeError("No trained model is available to save.")

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": MODEL,
        "companies": COMPANIES,
    }

    with path.open("wb") as file:
        pickle.dump(payload, file)

    return path


def load_model(
    model_path: str | Path = MODEL_FILE,
    dataset_path: str | Path = DATASET_FILE,
) -> Pipeline:
    """Load a saved model if present, otherwise train once from dataset.csv."""
    global MODEL, COMPANIES

    path = Path(model_path)
    if path.exists():
        with path.open("rb") as file:
            payload = pickle.load(file)

        MODEL = payload["model"]
        COMPANIES = payload["companies"]
        return MODEL

    return train_model(dataset_path=dataset_path, save=True)


def predict(text: str) -> dict[str, dict[str, float]]:
    """Return Up/Down/Neutral probabilities for each company."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    model = MODEL if MODEL is not None else load_model()
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("text became empty after cleaning.")

    probabilities = model.predict_proba([cleaned_text])[0]
    label_probabilities = dict(zip(model.classes_, probabilities))

    results: dict[str, dict[str, float]] = {}

    for company in COMPANIES:
        company_scores: dict[str, float] = {}
        total_score = 0.0

        for movement in MOVEMENTS:
            label = _make_label(company, movement)
            score = float(label_probabilities.get(label, 0.0))
            company_scores[movement] = score
            total_score += score

        # Normalize each company block so Up/Down/Neutral sums to 1.0.
        if total_score == 0.0:
            company_scores = {movement: round(1 / len(MOVEMENTS), 4) for movement in MOVEMENTS}
        else:
            company_scores = {
                movement: round(score / total_score, 4)
                for movement, score in company_scores.items()
            }

        results[company] = company_scores

    return results


__all__ = ["train_model", "save_model", "load_model", "predict"]
