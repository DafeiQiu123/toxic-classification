# Toxic Comment Classification with DistilBERT  
# 基于 DistilBERT 的多标签文本毒性分类完整报告

---

# 1. Project Overview / 项目概述

This project implements a full fine-tuning pipeline for multi-label toxicity classification using DistilBERT.

本项目实现了一个完整的预训练模型微调流程，使用 DistilBERT 进行 6 标签文本毒性分类任务。

Each comment can belong to multiple categories simultaneously:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

This is a **multi-label classification task**, not mutually exclusive classification.

---

# 2. Complete Pipeline / 完整模型流程

```
Raw Text
→ Tokenizer
→ Embedding
→ Transformer Encoder × 6
→ CLS Representation
→ Custom Classification Head
→ Sigmoid
→ BCEWithLogitsLoss
```

---

# 3. Data Processing / 数据处理

## 3.1 Raw Data Structure

Files:

- train.csv
- test.csv
- test_labels.csv

Columns:

```
id
comment_text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

We combine the 6 labels into a single 6-dimensional vector.

---

## 3.2 Tokenizer

Model:

```
distilbert-base-uncased
```

Tokenizer outputs:

- input_ids
- attention_mask

Example:

Text:

```
"You are stupid"
```

Tokenized:

```
[CLS] you are stupid [SEP]
```

Converted to:

```
input_ids:
[101, 2017, 2024, 4797, 102]

attention_mask:
[1, 1, 1, 1, 1]
```

Vocabulary size:

```
30522
```

Dynamic padding is applied inside DataLoader.

---

# 4. Embedding Layer / 嵌入层

DistilBERT embedding consists of:

- Token Embedding: (30522, 768)
- Position Embedding: (512, 768)

Formula:

```
Embedding = TokenEmbedding + PositionEmbedding
```

Output shape:

```
(Batch_size, seq_len, 768)
```

DistilBERT does NOT use segment embeddings.

---

# 5. Transformer Encoder (6 Layers)

DistilBERT contains 6 Transformer layers.

Each layer contains:

---

## 5.1 Multi-Head Self-Attention

Hidden size:

```
768
```

Number of heads:

```
12
```

Head dimension:

```
64
```

Because:

```
768 = 12 × 64
```

Each head has its own:

```
Q_i, K_i, V_i
```

Attention formula:

```
Attention(Q,K,V) = softmax(QKᵀ / √64) V
```

Shape flow:

```
(B, seq_len, 768)
→ reshape
(B, 12, seq_len, 64)
→ concat heads
(B, seq_len, 768)
```

---

## 5.2 Feed Forward Network (FFN)

Structure:

```
Linear (768 → 3072)
GELU
Linear (3072 → 768)
```

3072 = 4 × 768

Reason: expand feature space to improve nonlinear representation capacity.

---

# 6. CLS Representation

Final hidden states:

```
(B, seq_len, 768)
```

We extract:

```
CLS = hidden_states[:, 0, :]
```

Shape:

```
(B, 768)
```

Reason:

CLS attends to all tokens, summarizing the whole sentence.

---

# 7. Custom Classification Head

Instead of using HuggingFace’s built-in classification head, we defined:

```python
nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(768, 768),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(768, 6)
)
```

Output:

```
(B, 6 logits)
```

Then apply:

```
Sigmoid → probabilities (0~1)
```

---

# 8. Loss Function

We use:

```
BCEWithLogitsLoss
```

Reason:

- Multi-label classification
- Each label is independent
- Sigmoid + Binary Cross Entropy (numerically stable)

Softmax is not suitable because labels are not mutually exclusive.

---

# 9. Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Device | RTX 5070 Ti (CUDA 12.8) |
| Loss | BCEWithLogitsLoss |

---

# 10. Training Results

## Training Loss

| Epoch | Loss |
|--------|-------|
| 1 | 0.0463 |
| 2 | 0.0332 |
| 3 | 0.0265 |

Loss decreases steadily.

---

## Validation Performance

| Epoch | Micro-F1 | Macro-F1 |
|--------|-----------|-----------|
| 1 | 0.7735 | 0.5754 |
| 2 | 0.7840 | 0.6511 |
| 3 | 0.7901 | 0.6864 |

Best validation micro-F1:

```
0.7901
```

---

# 11. Test Evaluation (Ignoring -1 Labels)

test_labels.csv contains -1 values.

We ignore positions where:

```
label == -1
```

Test Results:

- micro_f1: 0.6662
- macro_f1: 0.6203

Per-label F1:

| Label | F1 |
|--------|------|
| toxic | 0.6597 |
| severe_toxic | 0.4361 |
| obscene | 0.6931 |
| threat | 0.6136 |
| insult | 0.6964 |
| identity_hate | 0.6230 |

Observation:

- Severe toxicity is hardest due to class imbalance.
- Obscene and insult perform better.

---

# 12. Engineering Challenges & Solutions

## 12.1 CUDA Compatibility Issue

### Problem
```text
no kernel image is available for execution
```

### Cause
The RTX 5070 Ti (compute capability `sm_120`) requires a newer CUDA-enabled PyTorch build. The installed PyTorch/CUDA stack did not include kernels compiled for `sm_120`, so CUDA kernel launches failed at runtime.

### Solution
Installed a PyTorch build compatible with CUDA 12.8 (cu128 wheel). After upgrading to the cu128-compatible PyTorch distribution, the kernel compatibility issue was resolved.

---

## 12.2 Dynamic Padding & GPU Memory Issue (OOM)

### Problem
Initially, I used `padding=True` during tokenization, which pads every sequence to a global fixed `max_length`. With a large dataset (~140k training samples), this expanded all sequences to the same length regardless of their true size. As a result, GPU memory usage increased significantly and training ran into out-of-memory (OOM) errors.

### Cause
Global fixed padding wastes memory and compute because:
- short inputs get padded with excessive padding tokens
- self-attention cost increases with sequence length (more tokens → heavier attention compute)
- batch tensors become larger, inflating GPU memory footprint
- training can become unstable under memory pressure

### Solution
Disabled padding during tokenization (`padding=False`) and implemented **dynamic padding** inside a custom `collate_fn` in the DataLoader using `tokenizer.pad()`. This pads each batch only to the **maximum sequence length within that batch**, rather than a global max.

Example:
```python
# tokenization stage
encoded = tokenizer(texts, padding=False, truncation=True)

# dynamic padding in DataLoader
def collate_fn(batch):
    return tokenizer.pad(batch, padding=True, return_tensors="pt")
```

### Benefits
- Reduced memory usage
- Faster training
- More efficient attention computation
- Improved training stability

---

## 12.3 Adam vs AdamW (Decoupled Weight Decay) for Transformer Fine-Tuning

### Problem
Initially, I used Adam with `weight_decay`:
```python
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
```
In PyTorch, this corresponds to Adam with **coupled L2 regularization**. Experimentally, this setup led to lower validation performance (micro-F1 and macro-F1) and less stable training.

### Cause
In PyTorch’s Adam implementation, `weight_decay` applies L2 regularization directly to the gradient (coupled weight decay). Because Adam uses adaptive scaling (moment estimates), the regularization term is also effectively scaled, which can distort the intended effect of weight decay. This is widely known to be suboptimal for Transformer fine-tuning.

### Solution
I ran an ablation across three optimizer setups:

1. **Pure Adam**: `Adam(..., weight_decay=0)`
2. **Adam + L2**: `Adam(..., weight_decay>0)` (PyTorch Adam applies *coupled* weight decay, effectively L2 regularization inside the gradient update)
3. **AdamW**: `AdamW(..., weight_decay>0)` (*decoupled* weight decay)

### Results (from training logs)

| Optimizer | Best micro-F1 | Best macro-F1 | Behavior across epochs |
|---|---:|---:|---|
| Adam + L2 (Adam, weight_decay>0) | **0.7200** | **0.3758** | Validation metrics **decreased** after epoch 1 |
| Pure Adam (weight_decay=0) | **0.7907** | **~0.637** | micro-F1 strong; macro-F1 improved slowly |
| AdamW | **0.7901** | **0.6864** | micro-F1 comparable to Adam; macro-F1 **consistently higher** |

### Interpretation
- **Micro-F1** is dominated by frequent labels, so **Pure Adam and AdamW are almost tied** (~0.79).
- **Macro-F1** weights each label equally, so it is much more sensitive to performance on rare/long-tail labels.
  - AdamW achieved a **much higher macro-F1 (0.686 vs ~0.637)**, indicating **better generalization on rare classes** and more balanced predictions.

---

# 13. Key Takeaways

Through this project:

- Implemented full fine-tuning pipeline
- Built custom classifier head
- Understood Transformer internals (Q, K, V, multi-head attention)
- Understood 768 = 12 × 64
- Understood FFN 768 → 3072 → 768
- Compared micro-F1 vs macro-F1
- Handled noisy labels (-1)
- Solved GPU compatibility issues
- Understood various optimizer 
