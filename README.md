# ai-transformer-decoder

Educational tutorial to learn to create Decoder-only transformer implemented from scratch in NumPy. Follows the GPT architecture from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper and Andrej Karpathy's ["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2408s).

---

## Architecture

![Transformer Decoder Architecture](architecture.png)

**Tokens → Token Embedding + Position Embedding → N × Transformer Block → Final Layer Norm → Output Projection → Logits**

Each transformer block contains two sub-layers with residual connections:

1. **Pre-Norm → Multi-Head Causal Self-Attention → Residual Add**
2. **Pre-Norm → Feed-Forward Network (expand 4×, ReLU, compress) → Residual Add**

---

## Model Config

| Parameter | Value | Description |
| --- | --- | --- |
| `vocab_size` | 65 | Unique tokens (characters) |
| `n_embd` | 64 | Embedding dimension per token |
| `n_head` | 4 | Parallel attention heads |
| `n_layer` | 4 | Stacked transformer blocks |
| `block_size` | 32 | Maximum context window |
| `head_size` | 16 | Embedding split per head (`n_embd // n_head`) |

---

## What's Implemented

**Self-Attention** — Each token produces Query, Key, and Value vectors. Dot product between Q and K gives attention scores, scaled by `√head_size`. Output is a weighted sum of V vectors.

**Causal Masking** — Upper-triangular entries in the attention matrix are set to `-inf` before softmax, preventing tokens from attending to future positions.

**Multi-Head Attention** — Embedding is split across 4 parallel heads. Each head computes attention independently, outputs are concatenated and projected back to full embedding dimension.

**Feed-Forward Network** — Two-layer MLP per token. Hidden layer expands to `4 × n_embd` with ReLU, then compresses back to `n_embd`.

**Residual Connections** — Input is added back after each sub-layer to maintain gradient flow across layers.

**Layer Normalization** — Pre-norm variant applied before attention and feed-forward sub-layers. Normalizes per-token vectors using learned `gamma` and `beta` parameters.

**Weight Initialization** — He initialization for linear layers. Token and position embeddings initialized with small random values (`0.02` scale).

---

## Run

```bash
python decoder.py
```

```
Input tokens shape:  (32,)
Output logits shape: (32, 65)
```

---

## References

- [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2408s)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)