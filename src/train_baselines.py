"""Leakage-aware TF-IDF baselines adapted from the thesis experiments."""

import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data import load_corpus, split_corpus


def build_models(seed: int) -> dict:
    return {
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Linear SVM": LinearSVC(random_state=seed),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a private CSV")
    parser.add_argument("--split", choices=["paragraph", "book"], default="paragraph")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = load_corpus(args.data)
    train, test = split_corpus(frame, args.split, args.test_size, args.seed)
    print(
        f"Split: {args.split} | train: {len(train)} | "
        f"evaluation: {len(test)} | authors: {frame['Author'].nunique()}"
    )
    for name, estimator in build_models(args.seed).items():
        # The vectorizer is fitted only on the train split via the pipeline.
        pipeline = make_pipeline(TfidfVectorizer(), estimator)
        pipeline.fit(train["Paragraph"], train["Author"])
        predicted = pipeline.predict(test["Paragraph"])
        print(
            f"{name}: accuracy={accuracy_score(test['Author'], predicted):.4f} "
            f"macro_f1={f1_score(test['Author'], predicted, average='macro', zero_division=0):.4f}"
        )


if __name__ == "__main__":
    main()
