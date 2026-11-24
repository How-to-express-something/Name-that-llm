"""Evaluate TF-IDF on held-out prompts; all vocabulary and IDF fitting is fold-local."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold

from baseline import make_model


def metrics(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def bootstrap_interval(frame, predicted, rng):
    # Resample prompts with replacement, keeping their model responses together.
    keys = [frame["prompt_id"], frame["model"]]
    correct = pd.Series(frame["model"].to_numpy() == predicted).groupby(keys).sum()
    support = frame.groupby(["prompt_id", "model"]).size()
    hits = correct.unstack(fill_value=0).to_numpy()
    counts = support.unstack(fill_value=0).to_numpy()
    n = len(counts)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=5000)
    samples = ((weights @ hits) / (weights @ counts)).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def run_evaluation(data_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False)
    text, actual = train["text"], train["model"].to_numpy()
    labels = np.unique(actual)
    prompts = np.sort(train["prompt_id"].unique())
    oof = train[["id", "prompt_id", "model"]].copy()
    splits = []
    for fold, (_, test_groups) in enumerate(KFold(5, shuffle=True, random_state=42).split(prompts), 1):
        held = train["prompt_id"].isin(prompts[test_groups]).to_numpy()
        splits.append((~held, held))
        oof.loc[held, "fold"] = fold
    oof["fold"] = oof["fold"].astype(int)
    variants = [
        ("stopwords_1", "word", (1, 1)),
        ("word_1", "word", (1, 1)),
        ("word_12", "word", (1, 2)),
        ("char_35", "char", (3, 5)),
    ]
    results = {"expected_random_balanced_accuracy": 1 / len(labels), "models": {}}
    for name, analyzer, ngrams in variants:
        predictions = np.empty(len(train), dtype=object)
        by_fold = []
        for fold, (training, held) in enumerate(splits, 1):
            model = make_model(analyzer, ngrams)
            if name == "stopwords_1":
                model.set_params(tfidf__vocabulary=sorted(ENGLISH_STOP_WORDS))
            model.fit(text[training], actual[training])
            pred = model.predict(text[held])
            predictions[held] = pred
            by_fold.append({"fold": fold, "rows": int(held.sum()), **metrics(actual[held], pred)})
        scores = metrics(actual, predictions)
        scores["folds"] = by_fold
        scores["balanced_accuracy_prompt_bootstrap_ci95"] = bootstrap_interval(
            train, predictions, np.random.default_rng(42)
        )
        results["models"][name] = scores
        oof[name] = predictions
        print(name, scores["balanced_accuracy"], flush=True)

    oof.to_csv(output_dir / "oof.csv", index=False)
    predicted = oof["word_12"].to_numpy()
    cm = confusion_matrix(actual, predicted, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis("actual").to_csv(output_dir / "confusion.csv")
    pd.DataFrame({
        "model": labels, "support": cm.sum(axis=1), "correct": cm.diagonal(),
        "recall": cm.diagonal() / cm.sum(axis=1),
    }).to_csv(output_dir / "per_class.csv", index=False)

    # Positive coefficient clues describe the fully fitted model used for submission.
    model = make_model().fit(text, actual)
    terms = model["tfidf"].get_feature_names_out()
    results["word_12_features"] = len(terms)
    features = []
    for label, coefficients in zip(model["svc"].classes_, model["svc"].coef_):
        for rank, index in enumerate(np.argsort(coefficients)[-8:][::-1], 1):
            features.append((label, rank, terms[index], coefficients[index]))
    pd.DataFrame(features, columns=["model", "rank", "term", "weight"]).to_csv(
        output_dir / "features.csv", index=False
    )
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_evaluation(args.data_dir, args.output_dir)
