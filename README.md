# Multilingual Author Detection

**Bachelor's thesis project | Python · NLP · text classification**

This project compares traditional machine learning, recurrent networks, and
transformer models for attributing short passages to one of six authors in a
Bangla and English book corpus. It grew from my B.Sc. Computer Science and
Engineering thesis, completed in 2024.

## What I did

- Collected and prepared paragraph-level samples from 27 books by six authors.
- Built TF-IDF baselines and experimented with LSTM, Bi-LSTM, BERT, BanglaBERT,
  DistilBERT, and XLM-RoBERTa.
- Compared model predictions using accuracy, F1 score, and confusion matrices.
- Revisited the original notebooks and made a cleaner training interface that
  keeps all dataset processing inside the training split.

## Thesis results

These are the **reported results of the original thesis experiments**, not
scores obtained by running the updated scripts in this repository:

| Model | Reported accuracy |
| --- | ---: |
| KNN | 88% |
| Random Forest | 90% |
| Decision Tree | 83% |
| SVM | 96% |
| Logistic Regression | 95% |
| LSTM | 82% |
| Bi-LSTM | 94% |
| DistilBERT | 95% |
| BanglaBERT | 98% |
| BERT | 96% |
| XLM-RoBERTa | 99% |

The original XLM-RoBERTa notebook records **514 correct predictions out of
519 validation samples (99.04%)** with a weighted F1 score of approximately
0.990. Its stratified split allocated 15% of the paragraph records to
validation, using random state 42.

### How to interpret the result

The validation set contained paragraphs from the **same collection of books**
as the training set. The score does not establish performance on unseen books
or authors. Each author in this corpus appears in only one language, so the
experiment also cannot isolate author style from language or demonstrate
cross-language attribution for the same author.

The original traditional-ML notebook fitted TF-IDF before the train/test
split; this can leak vocabulary information. The cleaned baseline script below
fits it only on training samples. Its new results will therefore not
necessarily match the historical table. The updated scripts offer a
book-held-out split as a stricter follow-up evaluation; **no book-held-out
score is claimed here**.

The report described 3,460 data points, while the archived CSV contains
**3,459 data rows plus a header**. No text or rows from that CSV are included
in this repository.

## Repository layout

- `src/data.py` — schema checks and paragraph/book split strategies.
- `src/train_baselines.py` — TF-IDF and five scikit-learn classifiers.
- `src/train_transformer.py` — configurable PyTorch/Hugging Face text classifier.
- `notebooks/original_cleaned/` — selected thesis notebooks, with outputs,
  metadata, local paths, and input excerpts removed. These preserve the
  experimental history; they have not been modernized or rerun.
- `data/README.md` — the expected private input format.

## Run the updated code

Install dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
```

Put your **own lawfully usable** CSV at a local path, or point the command to
your private copy. It must have `Author`, `Book Name`, and `Paragraph`
columns. Do not commit the CSV, model weights, or book scans. A paragraph
split mirrors the original XLM-R evaluation design; a book split holds out
whole books for a more demanding analysis.

```bash
python -m src.train_baselines --data path/to/private.csv --split paragraph
python -m src.train_baselines --data path/to/private.csv --split book
python -m src.train_transformer --data path/to/private.csv --model xlm-roberta-base --split paragraph
```

Use `--model bert-base-uncased` or `--model distilbert-base-uncased` to try
other architectures. A GPU is recommended for transformer training. These
commands refactor the original approach and have not reproduced the published
scores. They print aggregate metrics only and save no source text.

## Data and research artifacts

The original dataset contains book excerpts and is **private**. Its reuse for
research does not imply permission to redistribute it. The thesis report,
slides, clearance documents, raw notebooks with outputs, and any personal
records are also excluded.

## What I would test next

Measure performance on books absent from training, inspect errors per class,
check whether language alone predicts the label, and use licensed corpora
with authors represented in multiple languages before claiming cross-language
style recognition.
