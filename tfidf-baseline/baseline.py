"""Fit a text-only TF-IDF attribution model and generate a Kaggle submission."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def make_model(analyzer="word", ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        lowercase=True,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("svc", LinearSVC(C=1.0, dual=True, random_state=42)),
    ])


def run(data_dir, output_path, model_out=None):
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    train = pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(data_dir / "test.csv", dtype=str, keep_default_na=False)
    for frame in [train, test]:
        assert frame["id"].is_unique and frame["id"].str.len().min() > 0
        assert frame["text"].str.strip().str.len().min() > 0
    assert train["model"].nunique() == 18
    model = make_model()
    model.fit(train["text"], train["model"])
    submission = pd.DataFrame({"id": test["id"], "model": model.predict(test["text"])})
    assert submission["model"].isin(train["model"].unique()).all()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_parquet(output_path, engine="pyarrow", index=False)
    saved = pd.read_parquet(output_path)
    assert saved.columns.tolist() == ["id", "model"]
    assert saved["id"].tolist() == test["id"].tolist()
    assert saved["model"].tolist() == submission["model"].tolist()
    if model_out is not None:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_out, compress=3)
    print(f"{len(train)} train rows, {len(saved)} predictions written to {output_path}")
    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-out", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output, args.model_out)
