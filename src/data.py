"""Read private data and prepare transparent author-detection splits."""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = ("Author", "Book Name", "Paragraph")


def load_corpus(path: str | Path) -> pd.DataFrame:
    """Return non-empty, unique text records without printing their contents."""
    frame = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[(frame[list(REQUIRED_COLUMNS)] != "").all(axis=1)]
    frame = frame.drop_duplicates(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if frame.empty or frame["Author"].nunique() < 2:
        raise ValueError("The input needs at least two author classes and non-empty rows.")
    return frame


def split_corpus(
    frame: pd.DataFrame, strategy: str, test_size: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split paragraphs, or hold out entire books identified per author."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one.")
    if strategy == "paragraph":
        train, test = train_test_split(
            frame, test_size=test_size, random_state=seed, stratify=frame["Author"]
        )
    elif strategy == "book":
        # Pick held-out books within each class so all authors remain represented.
        groups = frame["Author"] + "::" + frame["Book Name"]
        rng = np.random.default_rng(seed)
        held_out: set[str] = set()
        for author, author_rows in frame.groupby("Author"):
            book_groups = groups.loc[author_rows.index].unique()
            if len(book_groups) < 2:
                raise ValueError(f"Book split requires two or more books for: {author}")
            n_holdout = min(len(book_groups) - 1, max(1, round(len(book_groups) * test_size)))
            held_out.update(rng.choice(book_groups, size=n_holdout, replace=False))
        in_test = groups.isin(held_out)
        train, test = frame.loc[~in_test], frame.loc[in_test]
    else:
        raise ValueError("strategy must be 'paragraph' or 'book'.")
    if not set(test["Author"]).issubset(set(train["Author"])):
        raise ValueError(
            "The held-out split has author classes with no training examples. "
            "Try another seed or use more books per author."
        )
    return train.reset_index(drop=True), test.reset_index(drop=True)
