import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
# step 1 使用 pandas 加载公开的文本毒性分类数据集（如 Jigsaw Toxic Comment Classification Challenge）。
# 工作并使用Hugging Face的 datasets 和 tokenizers 库，将文本数据转换成模型可以接受的格式（Input IDs, Attention Mask）
DATA_DIR = "../data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
SEED = 42

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_df["comment_text"] = train_df["comment_text"].fillna("")
    test_df["comment_text"] = test_df["comment_text"].fillna("")

    train_df["labels"] = train_df[LABEL_COLS].values.tolist()

    # shuffle + split
    train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    split = int(len(train_df) * 0.9)

    train_split_df = train_df.iloc[:split].copy() # 90% train
    val_split_df = train_df.iloc[split:].copy() # 10% validation

    # Pandas Dataframe -> Hugging Face Dataset
    train_ds = Dataset.from_pandas(
        train_split_df[["comment_text", "labels"]],
        preserve_index=False
    )

    val_ds = Dataset.from_pandas(
        val_split_df[["comment_text", "labels"]],
        preserve_index=False
    )

    test_ds = Dataset.from_pandas(
        test_df[["comment_text"]],
        preserve_index=False
    )

    # tokenizing
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False
        )
    # padding 先false, 省显存

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    print("============example")
    ex = train_ds[0]
    print("comment_text:", ex["comment_text"])
    print("labels:", ex["labels"])
    print("input_ids:", ex["input_ids"])
    print("attention_mask:", ex["attention_mask"])

    # to torch
    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    val_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )

    dsdict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

    out_dir = os.path.join(DATA_DIR, "processed_jigsaw_multilabel")
    dsdict.save_to_disk(out_dir)
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
