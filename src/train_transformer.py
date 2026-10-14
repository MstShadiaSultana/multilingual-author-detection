"""Updated transformer experiment based on the thesis XLM-R training loop."""

import argparse
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data import load_corpus, split_corpus


def encode(texts, labels, tokenizer, label_to_id, max_length):
    tokens = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return TensorDataset(
        tokens["input_ids"],
        tokens["attention_mask"],
        torch.tensor([label_to_id[label] for label in labels], dtype=torch.long),
    )


def evaluate(model, loader, device):
    model.eval()
    gold, predicted = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            gold.extend(labels.tolist())
            predicted.extend(output.logits.argmax(dim=-1).cpu().tolist())
    return accuracy_score(gold, predicted), f1_score(
        gold, predicted, average="weighted", zero_division=0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to a private CSV")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--split", choices=["paragraph", "book"], default="paragraph")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = load_corpus(args.data)
    train, test = split_corpus(frame, args.split, args.test_size, args.seed)
    authors = sorted(frame["Author"].unique())
    label_to_id = {label: index for index, label in enumerate(authors)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(authors)
    )
    train_data = encode(
        train["Paragraph"], train["Author"], tokenizer, label_to_id, args.max_length
    )
    test_data = encode(
        test["Paragraph"], test["Author"], tokenizer, label_to_id, args.max_length
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    print(
        f"Model: {args.model} | split: {args.split} | train: {len(train)} | "
        f"evaluation: {len(test)} | device: {device}"
    )
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            result = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += result.loss.item()
        accuracy, weighted_f1 = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch + 1}: mean_loss={total_loss / len(train_loader):.4f} "
            f"accuracy={accuracy:.4f} weighted_f1={weighted_f1:.4f}"
        )


if __name__ == "__main__":
    main()
