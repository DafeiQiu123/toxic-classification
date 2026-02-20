# evaluate.py
# 用 best_model.pt 预测 test.csv，输出 submission.csv
# 并与 test_labels.csv 对比（忽略 -1 label，只看 0/1）

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModel


LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# =========================
# 1) Same model as training.py
# =========================
class ToxicClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for distilbert-base
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (B, L, 768)
        cls_vector = last_hidden_state[:, 0, :]        # (B, 768)
        logits = self.classifier(cls_vector)           # (B, 6)
        return logits


# =========================
# 2) Args
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--ckpt_path", type=str, default="outputs_custom/best_model.pt")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)  # for converting prob->0/1 when scoring
    p.add_argument("--out_csv", type=str, default="submission.csv")
    return p.parse_args()


# =========================
# 3) Collate (dynamic padding)
# =========================
def make_collate_fn(tokenizer):
    def collate(batch):
        # batch is list of dicts: {"id":..., "input_ids":..., "attention_mask":..., "labels"(optional)}
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]

        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        out = {
            "id": [b["id"] for b in batch],
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }

        if "labels" in batch[0]:
            out["labels"] = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)

        return out
    return collate


# =========================
# 4) Build dataset from CSV (simple)
# =========================
def build_test_dataset(test_csv_path, tokenizer, max_len):
    df = pd.read_csv(test_csv_path)
    df["comment_text"] = df["comment_text"].fillna("")

    items = []
    for _, row in df.iterrows():
        text = row["comment_text"]
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding=False
        )
        items.append({
            "id": row["id"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        })
    return df, items


def attach_test_labels(items, test_labels_csv_path):
    """
    EN: add labels for scoring. keep -1 as is (ignore later).
    """
    lab = pd.read_csv(test_labels_csv_path)
    lab_map = {}
    for _, r in lab.iterrows():
        lab_map[r["id"]] = [int(r[c]) for c in LABEL_COLS]

    has = 0
    for it in items:
        if it["id"] in lab_map:
            it["labels"] = lab_map[it["id"]]
            has += 1
    print(f"[info] matched labels for {has}/{len(items)} test rows")
    return items


# =========================
# 5) Predict + Save submission + Optional scoring
# =========================
@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()
    all_ids = []
    all_probs = []
    all_labels = []  # optional

    for batch in tqdm(dataloader, desc="predict"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)  # (B,6)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_ids.extend(batch["id"])
        all_probs.append(probs)

        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    if len(all_labels) > 0:
        all_labels = np.concatenate(all_labels, axis=0)  # (N,6)
    else:
        all_labels = None

    return all_ids, all_probs, all_labels


def score_ignore_minus1(labels, probs, threshold=0.5):
    """
    EN: Compute F1 while ignoring positions where label == -1.
    CN: 忽略 label=-1 的位置，只在 0/1 上算指标。
    """
    preds = (probs >= threshold).astype(int)

    # mask valid positions
    valid_mask = (labels != -1)  # (N,6)
    y_true = labels[valid_mask].astype(int)
    y_pred = preds[valid_mask].astype(int)

    micro = f1_score(y_true, y_pred)

    # per-label F1 (ignore -1 per label)
    per_label = {}
    for j, name in enumerate(LABEL_COLS):
        m = labels[:, j] != -1
        if m.sum() == 0:
            per_label[name] = None
            continue
        per_label[name] = float(f1_score(labels[m, j].astype(int), preds[m, j].astype(int)))

    # macro over labels that exist
    vals = [v for v in per_label.values() if v is not None]
    macro = float(np.mean(vals)) if len(vals) else float("nan")

    return {"micro_f1": float(micro), "macro_f1": float(macro), "per_label_f1": per_label}


def main():
    args = parse_args()

    test_csv = os.path.join(args.data_dir, "test.csv")
    test_labels_csv = os.path.join(args.data_dir, "test_labels.csv")

    assert os.path.exists(test_csv), f"test.csv not found at: {test_csv}"
    assert os.path.exists(args.ckpt_path), f"checkpoint not found: {args.ckpt_path}"

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # build dataset
    test_df, items = build_test_dataset(test_csv, tokenizer, args.max_len)

    # attach labels if exists
    has_labels = os.path.exists(test_labels_csv)
    if has_labels:
        items = attach_test_labels(items, test_labels_csv)
    else:
        print("[warn] test_labels.csv not found; will only generate submission.csv")

    # dataloader
    collate_fn = make_collate_fn(tokenizer)
    loader = DataLoader(items, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # model + load weights
    model = ToxicClassifier(model_name=args.model_name, num_labels=6).to(device)
    state = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    print(f"[info] loaded weights from {args.ckpt_path}")

    # predict
    ids, probs, labels = predict(model, loader, device)

    # save submission.csv (id + 6 prob columns)
    sub = pd.DataFrame({"id": ids})
    for j, name in enumerate(LABEL_COLS):
        sub[name] = probs[:, j]

    sub.to_csv(args.out_csv, index=False)
    print(f"[saved] submission to: {args.out_csv}")
    print(sub.head(3))

    # score if labels exist
    if labels is not None:
        metrics = score_ignore_minus1(labels, probs, threshold=args.threshold)
        print("\n== Metrics (ignore label=-1) ==")
        print("micro_f1:", metrics["micro_f1"])
        print("macro_f1:", metrics["macro_f1"])
        print("per_label_f1:")
        for k, v in metrics["per_label_f1"].items():
            print(f"  {k}: {v}")

        # extra: how many valid labels per class
        valid_counts = {name: int((labels[:, j] != -1).sum()) for j, name in enumerate(LABEL_COLS)}
        print("\nvalid label counts (excluding -1):", valid_counts)


if __name__ == "__main__":
    main()
