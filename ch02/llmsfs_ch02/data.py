from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken


class GPTDatasetV1(Dataset):
    """
    Sliding-window next-token dataset:
    input = token_ids[i : i+L]
    target = token_ids[i+1 : i+L+1]
    """
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        if len(token_ids) <= max_length:
            raise ValueError("Need at least max_length+1 tokens to build (input, target) pairs")

        for i in range(0, len(token_ids) - max_length, stride):
            x = token_ids[i : i + max_length]
            y = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(x, dtype=torch.long))
            self.target_ids.append(torch.tensor(y, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 4
    max_length: int = 256
    stride: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0


def create_dataloader_v1(text: str, cfg: DataLoaderConfig) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    ds = GPTDatasetV1(text, tokenizer, max_length=cfg.max_length, stride=cfg.stride)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
    )