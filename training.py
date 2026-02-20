# training_custom.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



# self defined classifier
class ToxicClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        last_hidden_state = outputs.last_hidden_state  # (B, L, 768)

        cls_vector = last_hidden_state[:, 0, :]        # (B, 768)

        logits = self.classifier(cls_vector)           # (B, 6)

        return logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../data/processed_jigsaw_multilabel")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="outputs_custom")
    return p.parse_args()

# padding
def make_collate_fn(tokenizer):
    def collate(batch):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch]).float()

        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels
        }
    return collate


# evaluation
@torch.no_grad()
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()

    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    preds = (probs >= threshold).astype(int)

    micro = f1_score(labels.reshape(-1), preds.reshape(-1))
    macro = f1_score(labels, preds, average="macro")

    return {"micro_f1": float(micro), "macro_f1": float(macro)}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) load dataset
    dsdict = load_from_disk(args.data_dir)
    train_ds = dsdict["train"]
    val_ds = dsdict["validation"]

    # 2) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # 3) dataloaders
    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4) model
    model = ToxicClassifier(
        model_name=args.model_name,
        num_labels=6
    ).to(device)

    # 5) optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 6) loss
    loss_fn = nn.BCEWithLogitsLoss()

    best_micro = -1.0

    # epoch start

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / step)

        epoch_loss = running_loss / len(train_loader)
        print(f"[epoch {epoch}] training loss: {epoch_loss:.4f}")

        print("Training finished for epoch", epoch)

        metrics = evaluate(model, val_loader, device, args.threshold)
        print(f"[epoch {epoch}] val metrics:", metrics)

        if metrics["micro_f1"] > best_micro:
            best_micro = metrics["micro_f1"]
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"[saved] best model (micro_f1={best_micro:.4f})")

    print("\nDone. Best micro_f1 =", best_micro)


if __name__ == "__main__":
    main()
