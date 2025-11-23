# Chapter 2: Working with Text Data - Key Learning Steps

## 1. Tokenization
**Goal:** Convert raw text into smaller units (tokens) that a machine can process.

*   **Basic Approach (Regex):**
    *   We started by using `re.split` to separate text by words, punctuation, and whitespace.
    *   We built a custom `SimpleTokenizerV2` class to handle encoding (text → IDs) and decoding (IDs → text).
*   **Production Approach (BPE):**
    *   We switched to **Byte Pair Encoding (BPE)** using the `tiktoken` library (specifically the `gpt2` encoding).
    *   *Why?* BPE efficiently handles unknown words by breaking them into subword units and maintains a fixed vocabulary size (e.g., 50,257 for GPT-2).

## 2. Special Tokens
**Goal:** Handle edge cases and context boundaries.

*   **`<|unk|>`**: Used for "unknown" words not found in the vocabulary (though BPE largely mitigates this).
*   **`<|endoftext|>`**: A marker used to signal the end of a document or to separate distinct text sources within a single data stream.

## 3. Data Sampling (Sliding Window)
**Goal:** Structure the data for "Next-Token Prediction" (Self-Supervised Learning).

*   **Input vs. Target:**
    *   **Input (`x`):** A sequence of tokens of length `max_length`.
    *   **Target (`y`):** The exact same sequence shifted to the right by **one position**.
    *   *Concept:* Given tokens `t_1` to `t_n`, the model learns to predict `t_{n+1}`.
*   **Sliding Window:**
    *   We move a window across the text to create multiple examples.
    *   **Stride:** Determines how many steps the window moves. A stride smaller than `max_length` creates overlapping samples, providing more training data.

## 4. The Dataset & DataLoader
**Goal:** Efficiently load and batch data for PyTorch training.

*   **`GPTDatasetV1`**:
    *   Inherits from PyTorch's `Dataset` class.
    *   Tokenizes the entire text into a single long sequence.
    *   Slices this sequence into `(input, target)` pairs based on the sliding window logic.
*   **`DataLoader`**:
    *   Batches multiple samples together (e.g., creating tensors of shape `[batch_size, max_length]`).
    *   Handles shuffling to improve training stability.

## 5. Embeddings
**Goal:** Convert integer Token IDs into dense vectors containing semantic information.

*   **Token Embeddings:**
    *   Implemented using `torch.nn.Embedding(vocab_size, output_dim)`.
    *   Each integer ID serves as an index to look up a trainable vector.
*   **Positional Embeddings:**
    *   *Problem:* The Transformer architecture has no inherent sense of order/sequence.
    *   *Solution:* We added a second embedding layer `torch.nn.Embedding(context_length, output_dim)` representing absolute positions (0, 1, 2...).
*   **Final Input Representation:**
    *   The input to the model is the element-wise sum:
    *   `Input_Embeddings = Token_Embeddings + Positional_Embeddings`