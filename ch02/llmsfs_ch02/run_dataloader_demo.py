from __future__ import annotations

import tiktoken

from ch02.llmsfs_ch02.data import DataLoaderConfig, create_dataloader_v1
from ch02.llmsfs_ch02.io_utils import data_path, read_text


def main() -> None:
    text = read_text(data_path("the-verdict.txt"))

    # Make the demo small and deterministic
    cfg = DataLoaderConfig(
        batch_size=2,
        max_length=8,
        stride=4,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    dl = create_dataloader_v1(text, cfg)

    enc = tiktoken.get_encoding("gpt2")

    print("Demo config:", cfg)
    print("num batches:", len(dl))

    # Show a couple batches + decode the first sample for intuition
    for batch_idx, (x, y) in enumerate(dl):
        print(f"\nBatch {batch_idx}")
        print("x shape:", tuple(x.shape))
        print("y shape:", tuple(y.shape))

        # Print the first sequence of the batch as token ids
        x0 = x[0].tolist()
        y0 = y[0].tolist()
        print("x[0] token ids:", x0)
        print("y[0] token ids:", y0)

        # Decode x[0] and the "next token" target y[0][-1]
        # Note: decoding BPE tokens will often include leading spaces; that's expected.
        print("decoded x[0]:", repr(enc.decode(x0)))
        print("decoded next token (y[0][-1]):", repr(enc.decode([y0[-1]])))

        if batch_idx >= 1:
            break


if __name__ == "__main__":
    main()