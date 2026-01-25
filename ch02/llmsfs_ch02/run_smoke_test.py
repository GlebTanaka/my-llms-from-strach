from __future__ import annotations

import torch
import tiktoken

from ch02.llmsfs_ch02.data import DataLoaderConfig, create_dataloader_v1
from ch02.llmsfs_ch02.embeddings import position_embedding_table, token_embedding_table
from ch02.llmsfs_ch02.io_utils import data_path, read_text


def main() -> None:
    text_path = data_path("the-verdict.txt")
    text = read_text(text_path)

    cfg = DataLoaderConfig(batch_size=2, max_length=8, stride=4, shuffle=False)
    dl = create_dataloader_v1(text, cfg)

    x, y = next(iter(dl))
    print("x shape:", tuple(x.shape))  # (batch, seq)
    print("y shape:", tuple(y.shape))  # (batch, seq)

    # Embedding demo (GPT-2 vocab size)
    vocab_size = 50257
    d_model = 256
    tok_emb = token_embedding_table(vocab_size, d_model)
    pos_emb = position_embedding_table(cfg.max_length, d_model)

    token_vecs = tok_emb(x)  # [batch, seq, d_model]
    pos_vecs = pos_emb(torch.arange(cfg.max_length))  # [seq, d_model]
    input_emb = token_vecs + pos_vecs

    print("token_vecs shape:", tuple(token_vecs.shape))
    print("pos_vecs shape:", tuple(pos_vecs.shape))
    print("input_emb shape:", tuple(input_emb.shape))

    # Optional: show tokenizer token count quickly
    enc = tiktoken.get_encoding("gpt2")
    print("num tokens (gpt2):", len(enc.encode(text)))


if __name__ == "__main__":
    main()