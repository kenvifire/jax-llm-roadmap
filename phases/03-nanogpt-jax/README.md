# 03 — nanoGPT in JAX

## Goal

Scale the Phase 02 model up to a real GPT-2 small (124M params) training
run on a single TPU, with proper checkpointing and data loading instead of
the toy in-memory setup from Phase 02.

## Study

- Karpathy — nanoGPT: https://github.com/karpathy/nanoGPT (read the model
  and training loop even though it's PyTorch — the architecture and
  training recipe carry over directly)
- Orbax (checkpointing): https://github.com/google/orbax
- Grain (data loading): https://github.com/google/grain

## Files

This phase has no starter script in the repo yet — build it here as you go.
Suggested layout once you start:

- `model.py` — GPT-2-small-sized config on top of the Phase 02 `GPT` class
  (or a copy of it), with rotary/learned position embeddings as you prefer.
- `data.py` — a Grain `DataLoader` over a tokenized text corpus.
- `train.py` — the training loop, with Orbax checkpoint saving/restoring
  on an interval.

## Do

- [ ] Pick a dataset (e.g. OpenWebText subset, or anything you can tokenize
      with `tiktoken`) and get it into a Grain-loadable format.
- [ ] Scale the Phase 02 `GPTConfig` up to GPT-2 small: `n_layer=12`,
      `n_head=12`, `n_embd=768`, `block_size=1024`.
- [ ] Get a training run going on a single TPU chip (Kaggle/Colab TPU is
      fine to start).
- [ ] Add Orbax checkpoint saving every N steps, and confirm you can
      restore from a checkpoint and continue training.
- [ ] Commit a short write-up of the run (loss curve, wall-clock time,
      steps/sec) alongside the code.

## Check yourself

1. What does Grain give you over a plain Python generator for feeding
   batches into a `jit`'d training step? Why does batch-shape consistency
   matter more here than it did for the small char-level dataset in
   Phase 02?
2. What exactly does an Orbax checkpoint need to contain to let you resume
   training *exactly* where you left off — model params alone, or more?
3. At GPT-2-small scale, what's the first thing likely to limit your step
   time on a single TPU chip: compute, host-to-device transfer, or data
   loading? How would you check which one it actually is?
