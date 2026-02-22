import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

DATA_DIR = "../data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
SEED = 42

LABEL_COLS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def main():
    # 1) 用 HF datasets 直接读 CSV（不用 pandas）
    raw = load_dataset(
        "csv",
        data_files={"train": TRAIN_CSV, "test": TEST_CSV},
    )

    # 2) train 划分 validation（HF 自带）
    split = raw["train"].train_test_split(test_size=0.1, seed=SEED)
    dsdict = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": raw["test"],
    })

    # 3) 构造 6-label 向量：labels = [toxic, ..., identity_hate]
    def build_labels(example):
        example["labels"] = [int(example[c]) for c in LABEL_COLS]
        return example

    dsdict["train"] = dsdict["train"].map(build_labels)
    dsdict["validation"] = dsdict["validation"].map(build_labels)

    # 4) tokenizer（transformers 里自带 fast tokenizer；底层是 tokenizers）
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(
            batch["comment_text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,  # 后续 DataLoader 动态 padding
        )

    dsdict["train"] = dsdict["train"].map(tokenize_fn, batched=True, remove_columns=[])
    dsdict["validation"] = dsdict["validation"].map(tokenize_fn, batched=True, remove_columns=[])
    dsdict["test"] = dsdict["test"].map(tokenize_fn, batched=True, remove_columns=[])

    # 5) 只保留你训练/推理需要的字段（让数据更干净）
    keep_train = ["input_ids", "attention_mask", "labels"]
    keep_test = ["input_ids", "attention_mask"]

    dsdict["train"] = dsdict["train"].remove_columns(
        [c for c in dsdict["train"].column_names if c not in keep_train]
    )
    dsdict["validation"] = dsdict["validation"].remove_columns(
        [c for c in dsdict["validation"].column_names if c not in keep_train]
    )
    dsdict["test"] = dsdict["test"].remove_columns(
        [c for c in dsdict["test"].column_names if c not in keep_test]
    )

    # 6) 设置 torch 格式（训练时直接 tensor）
    dsdict["train"].set_format(type="torch", columns=keep_train)
    dsdict["validation"].set_format(type="torch", columns=keep_train)
    dsdict["test"].set_format(type="torch", columns=keep_test)

    # 7) 检查样本
    sample = dsdict["train"][0]
    print("sample keys:", sample.keys())
    print("labels:", sample["labels"])
    print("len(input_ids):", len(sample["input_ids"]))

    # 8) 保存
    out_dir = os.path.join(DATA_DIR, "processed_jigsaw_multilabel_hf")
    dsdict.save_to_disk(out_dir)
    print("Saved to:", out_dir)

if __name__ == "__main__":
    main()
