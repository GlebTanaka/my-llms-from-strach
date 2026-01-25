# Chapter 2 — Working with Text Data

## Key ideas

### 1) Tokenization is a design choice
**Goal:** convert raw text into a sequence of discrete units (tokens) that can be mapped to integers.

- **Regex / “word-ish” tokenization**
    - Easy to implement and reason about.
    - Good for learning what tokenization *does*.
    - Weakness: out-of-vocabulary words and edge cases (punctuation, casing, unicode, etc.).

- **BPE tokenization (production-style)**
    - Breaks text into **subword** units, so it handles unseen words much better.
    - Uses a fixed vocabulary size (e.g., GPT‑2 tokenizer vocab size is 50,257).
    - Practical takeaway: when you train GPT-like models, BPE-style tokenization is the default.

### 2) Token IDs and vocab are the bridge to neural nets
Once you have tokens, you build a **token → id** mapping so you can represent text as integers.

- Determinism matters: building the vocab in a stable way makes experiments reproducible.
- Many pipelines add **special tokens** to represent boundaries and unknowns.

### 3) Next-token prediction dataset creation (sliding window)
Transformers learn from a simple self-supervised objective:

- **Input**: a fixed-length sequence of token IDs `x = [t_i, ..., t_{i+L-1}]`
- **Target**: the same sequence shifted by one `y = [t_{i+1}, ..., t_{i+L}]`

A **sliding window** creates many overlapping training examples from one long stream of tokens.

- `max_length` controls context length (`L`)
- `stride` controls how much overlap you get (smaller stride → more overlap → more samples)

### 4) DataLoader = batching + iteration for training
- `Dataset` builds `(x, y)` pairs.
- `DataLoader` batches them into tensors of shape:
    - `x`: `[batch_size, max_length]`
    - `y`: `[batch_size, max_length]`

This is the format used later for training a GPT model.

### 5) Embeddings: integers → vectors
Neural networks don’t operate on token IDs directly.

- **Token embedding**: lookup table `Embedding(vocab_size, d_model)`  
  turns token IDs into vectors → `[batch, seq, d_model]`
- **Positional embedding**: another lookup table `Embedding(context_length, d_model)`  
  gives each position `0..L-1` a vector → `[seq, d_model]`
- **Final model input**:  
  `input_embeddings = token_embeddings + positional_embeddings` (broadcast across batch)

Practical takeaway: transformers need *some* notion of position; in this chapter you used **learned absolute positions**.

---

## Deviations

Compared to a notebook-only workflow:

- I keep the chapter **self-contained** and runnable as a module (from repo root).
- I extracted the reusable parts into a small chapter package (`llmsfs_ch02`) and added runnable demos:
    - a smoke test to validate shapes and the end-to-end pipeline
    - small demo runners to reinforce tokenization + dataloader intuition
- I kept older exploratory code around as reference, but treat the package code as the “clean” implementation.

---

## Questions I still have (good things to revisit)

- Tokenization:
    - What are the tradeoffs between “keep punctuation as tokens” vs “strip punctuation”?
    - For BPE tokenization, why do some decoded tokens include leading spaces (and why is that not a bug)?

- Dataset/windowing:
    - How do `stride` and `max_length` affect training speed vs quality?
    - How should we handle **document boundaries** in training streams (when to insert an end-of-text token)?

- Embeddings:
    - What breaks when the sequence length exceeds `context_length` with learned absolute position embeddings?
    - What alternatives exist (sinusoidal / rotary), and what do they change in the model code?

---

## Next chapter hooks (what this enables in Chapter 3)

Chapter 2 builds the **input pipeline** that attention will consume:

- You can now produce `input_embeddings` shaped `[batch, seq, d_model]`.
- Next, attention mechanisms will:
    - project those embeddings into queries/keys/values
    - compute attention weights across the sequence
    - produce contextualized token representations

A helpful mental bridge:
- **Ch02** answers: “How do I turn text into tensors?”
- **Ch03** answers: “How do tensors learn relationships across the sequence?”