# Private input data

This folder intentionally contains no dataset.

The training scripts expect a CSV with the columns:

| Column | Meaning |
| --- | --- |
| `Author` | Class label for the writer |
| `Book Name` | Source book identifier for group splitting |
| `Paragraph` | Text sample |

Supply a lawful private CSV via `--data /path/to/private.csv`. No book
passages or samples from the original corpus are included here. Public users
may use their own licensed text with the same schema.
