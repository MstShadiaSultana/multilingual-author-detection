# Research notebook archive

These seven notebooks retain the thesis-era code cells as a record of the
experiments. Display outputs, Markdown, private Colab/Kaggle paths and drive
mount commands were removed. CSV reads now use
`../../data/private/author_paragraphs.csv` relative to this folder; set that
path locally if you want to inspect them. No dataset is in this repository.

**These are archival notebooks, not supported or rerun workflows.** Some
cells relied on older libraries or a particular execution order. The original
DistilBERT notebook specifies a DistilBERT checkpoint but uses BERT-specific
classes; it should not be treated as an executable DistilBERT implementation.
The original TF-IDF notebook fits the vectorizer before splitting and therefore
has a leakage risk. The original XLM-R notebook has an inconsistent
`best_valid_loss` / `best_val_loss` checkpoint variable. Use the updated
`src/` scripts for a cleaner starting point.
