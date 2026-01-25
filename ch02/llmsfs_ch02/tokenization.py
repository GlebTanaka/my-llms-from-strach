from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


_DEFAULT_SPLIT_PATTERN = r"([,.:;?_!\"()\']|--|\s)"


def split_keep_punct(text: str, pattern: str = _DEFAULT_SPLIT_PATTERN) -> List[str]:
    """
    Splits text into tokens while keeping punctuation as separate tokens.
    Whitespace is used as a separator and is not kept.
    """
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def build_vocab(tokens: Iterable[str]) -> Dict[str, int]:
    """
    Deterministic vocab: sorted unique tokens -> ids.
    """
    vocab = sorted(set(tokens))
    return {tok: i for i, tok in enumerate(vocab)}


@dataclass(frozen=True)
class SpecialTokens:
    endoftext: str = "<|endoftext|>"
    unk: str = "<|unk|>"


def add_special_tokens(token_to_id: Dict[str, int], special: SpecialTokens = SpecialTokens()) -> Dict[str, int]:
    """
    Returns a new token_to_id with special tokens appended at the end.
    """
    out = dict(token_to_id)
    next_id = len(out)
    if special.endoftext not in out:
        out[special.endoftext] = next_id
        next_id += 1
    if special.unk not in out:
        out[special.unk] = next_id
    return out


class SimpleTokenizerV1:
    """
    Regex tokenizer that assumes every token exists in the vocabulary (no unknown handling).
    """
    def __init__(self, token_to_id: Dict[str, int], pattern: str = _DEFAULT_SPLIT_PATTERN):
        self.str_to_int = dict(token_to_id)
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}
        self.pattern = pattern

    def encode(self, text: str) -> List[int]:
        tokens = split_keep_punct(text, self.pattern)
        return [self.str_to_int[t] for t in tokens]

    def decode(self, ids: List[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r"\s+([,.?!" + r"\"()\'])", r"\1", text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """
    Same tokenizer, but maps unknown tokens to <|unk|>.
    """
    def __init__(self, token_to_id: Dict[str, int], pattern: str = _DEFAULT_SPLIT_PATTERN, unk_token: str = "<|unk|>"):
        super().__init__(token_to_id, pattern)
        if unk_token not in self.str_to_int:
            raise ValueError(f"unk token {unk_token!r} not found in vocab")
        self.unk_token_id = self.str_to_int[unk_token]

    def encode(self, text: str) -> List[int]:
        tokens = split_keep_punct(text, self.pattern)
        return [self.str_to_int.get(t, self.unk_token_id) for t in tokens]