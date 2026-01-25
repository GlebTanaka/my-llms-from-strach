from __future__ import annotations

from ch02.llmsfs_ch02.io_utils import data_path, read_text
from ch02.llmsfs_ch02.tokenization import (
    SimpleTokenizerV1,
    SimpleTokenizerV2,
    add_special_tokens,
    build_vocab,
    split_keep_punct,
)


def main() -> None:
    text = read_text(data_path("the-verdict.txt"))

    # Build a deterministic vocab from the chapter text
    tokens = split_keep_punct(text)
    token_to_id = build_vocab(tokens)
    token_to_id = add_special_tokens(token_to_id)

    print("vocab size (incl. specials):", len(token_to_id))

    # Demonstrate encode/decode (known tokens)
    tok_v1 = SimpleTokenizerV1(token_to_id)
    example = "Hello, world!"
    # Note: v1 will KeyError if tokens are missing, so we use v2 for robust demo text.
    tok_v2 = SimpleTokenizerV2(token_to_id)

    ids = tok_v2.encode(example)
    roundtrip = tok_v2.decode(ids)
    print("\nExample text:", example)
    print("Encoded ids:", ids)
    print("Decoded text:", roundtrip)

    # Demonstrate unknown handling explicitly
    unknown_example = "hello nonexistentword world"
    ids2 = tok_v2.encode(unknown_example)
    print("\nUnknown example:", unknown_example)
    print("Encoded ids:", ids2)
    print("Decoded text:", tok_v2.decode(ids2))

    # Show the first N tokens from the corpus
    print("\nFirst 30 tokens from corpus:")
    print(tokens[:30])


if __name__ == "__main__":
    main()