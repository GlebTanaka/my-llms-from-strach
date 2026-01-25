from __future__ import annotations

import torch
import torch.nn as nn


def token_embedding_table(vocab_size: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)


def position_embedding_table(context_length: int, d_model: int) -> nn.Embedding:
    return nn.Embedding(num_embeddings=context_length, embedding_dim=d_model)


def add_token_and_position_embeddings(
        token_embeddings: torch.Tensor,  # [batch, seq, d_model]
        pos_embeddings: torch.Tensor,    # [seq, d_model] or [1, seq, d_model]
) -> torch.Tensor:
    return token_embeddings + pos_embeddings