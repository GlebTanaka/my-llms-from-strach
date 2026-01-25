# agents.md — Project working agreement (me + AI assistant)

## Scope / chapter roadmap
This repo follows “Building LLMs from scratch” starting at Chapter 2:

- Ch02: Working with text data
- Ch03: Coding attention mechanisms
- Ch04: Implementing a GPT model from scratch (text generation)
- Ch05: Pretraining on unlabeled data
- Ch06: Fine-tuning for classification
- Ch07: Fine-tuning to follow instructions

Chapter 1 (intro/setup) is intentionally not included here.

## Primary workflow (per chapter)
1. Follow along in notebooks to learn and validate results.
2. After finishing the chapter, consolidate the concepts into one or more `.py` files:
   - More structure (functions/classes)
   - Heavier comments explaining *why*
   - Minimal “notebook noise” (prints, quick experiments)

## Source of truth
- Notebooks = learning trail and experiments
- `.py` files = curated reference implementation for review/reuse
- If they disagree: `.py` represents “final” unless explicitly marked experimental.

## Chapter folder conventions
Each `chXX/` may contain:
- `notebooks/`: follow-along notebooks and experiment notebooks
- `src/`: curated Python implementations for the chapter
- `notes/`: summary + deviations + results

(Exact folders may start simple and become more structured later, but keep the intent.)

## Deviations / additions policy
Deviations are encouraged if they strengthen understanding.
Rule: document deviations in the chapter summary under “Deviations & Notes”.

Examples of acceptable deviations:
- extra tests/sanity checks
- clearer refactors in `.py`
- alternative implementations (clearly labeled)
- extra exercises after finishing the chapter

## Shared code policy (important later)
Until Ch04 is complete:
- Prefer chapter-local code (duplication is okay for learning).

After Ch04:
- We may extract stable utilities (tokenization/data/model components) into a shared module/package.

## AI assistant guidance
- Make minimal, reversible changes.
- Don’t reorder notebook cells unless asked.
- Preserve behavior unless explicitly asked to change behavior.
- Prefer clarity over cleverness; add comments explaining reasoning.

## Running code (module-style)
- Run modules from repo root using:
   - `python -m chXX.<chapter_package>.run_train`
- Do not rely on running files by path (avoids import issues).
- Each runnable entry point is a small `run_*.py` that calls functions in `train.py`, etc.

## Naming
- Chapter folders: `ch02`, `ch03`, ... `ch07`
- Chapter code package inside each chapter: `llmsfs_chXX`
   - Example: `ch04/llmsfs_ch04/`

## Isolation
- Chapters are self-contained.
- No imports across chapters until after Chapter 4.

## Running (module-style)
- Run from repo root:
   - `python -m chXX.llmsfs_chXX.run_*`
