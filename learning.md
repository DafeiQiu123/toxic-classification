# this document is personal learning note

1. data
comment_text

2. tokenizer
input_ids + attention_mask(padding)
input_ids from distilbert-base-uncased - vocab.txt

3. embedding layer
token embedding + position embedding + (Segement (not in bert))
 (30522,768)       (512,768)
 -> (Batch size, seq_len, 768)
 
4. Transformer Encoder * 6 (Distilbert; *12 Bert)

Multi-Head Self-Attention
12 different Q,K,V -> (B, seq_len, 768) ; softmax(QKᵀ / √64) V
=> (B, 12, seq_len, 64) ; concat heads
=> (B,seq_len,768)
12 heads * 64 dim 

Feed Forward
768->3072->768
Linear (768 → 3072)
GELU
Linear (3072 → 768)

5. Final Hidden States -> [CLS] Vector
(B,seq_len,768) -> (B,768)
CLS ↔ 所有 token 互相注意
hence taking only cls vector is enough

6. Linear Layer for our classification specifically
768 -> 6 logits -> Sigmoid

7. Loss use BCEWithLogitsLoss (sigmoid + binary cross entropy)