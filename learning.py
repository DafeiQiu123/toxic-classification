import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    model_name = "distilbert-base-uncased"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    print("\n===== Model Basic Info =====")
    print("Hidden size:", model.config.hidden_size)
    print("Vocab size:", model.config.vocab_size)
    print("Max position embeddings:", model.config.max_position_embeddings)

    # 1️⃣ 打印 embedding 矩阵形状
    word_embed = model.distilbert.embeddings.word_embeddings.weight
    pos_embed = model.distilbert.embeddings.position_embeddings.weight

    print("\n===== Embedding Shapes =====")
    print("Token Embedding shape:", word_embed.shape)  # (30522, 768)
    print("Position Embedding shape:", pos_embed.shape)  # (512, 768)

    # 2️⃣ 查看某个 token 的 embedding
    token_id = 2017  # 2017 对应 "you"
    token_str = tokenizer.convert_ids_to_tokens(token_id)

    print("\n===== Inspect Token Embedding =====")
    print("Token ID:", token_id)
    print("Token string:", token_str)
    print("First 10 dims of embedding:")
    print(word_embed[token_id][:10])

    # 3️⃣ 查看某个 position 的 embedding
    pos_id = 0
    print("\n===== Inspect Position Embedding =====")
    print("Position:", pos_id)
    print("First 10 dims of position embedding:")
    print(pos_embed[pos_id][:10])


if __name__ == "__main__":
    main()

