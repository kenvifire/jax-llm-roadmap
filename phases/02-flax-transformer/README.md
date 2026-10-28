# 02 — Flax + transformer

## Goal

Move from raw JAX to Flax NNX, and build a small GPT with hand-written
causal self-attention so the mechanics aren't hidden behind a library
module. Train it as a char-level language model with Optax.

## Study

- Flax NNX docs: https://flax.readthedocs.io/en/latest/nnx/index.html
- Optax docs: https://optax.readthedocs.io/
- Karpathy — Neural Networks: Zero to Hero (the GPT videos in particular):
  https://karpathy.ai/zero-to-hero.html
- Vaswani et al., *Attention Is All You Need* (see [RESOURCES.md](../../RESOURCES.md))

## Files

- `model.py` — GPT in Flax NNX: hand-written causal self-attention (no
  `nnx.MultiHeadAttention`), pre-norm blocks, learned position embeddings,
  weight tying via `wte.attend`, and a `generate()` method.
- `train_char.py` — char-level training loop. Uses `input.txt` if you drop
  one in this directory, otherwise falls back to the repo's own
  `.py`/`.md` files so it runs with zero setup.

## Do

- [ ] Run `model.py` directly and confirm the forward pass and `generate`
      smoke tests pass.
- [ ] Run `train_char.py` and watch train/val loss drop over 300 steps.
- [ ] Drop in a real `input.txt` (e.g. tinyshakespeare) and re-run — compare
      the sampled text quality against the repo-text fallback.
- [ ] Increase `N_LAYER` / `N_EMBD` in `train_char.py` and see how much
      lower you can get the validation loss in the same number of steps.

## Check yourself

1. Why is `wte.attend(x)` used for the output head instead of a separate
   `nnx.Linear(n_embd, vocab_size)`? What parameters does this save, and
   what assumption does it make about the embedding and output spaces?
2. In `CausalSelfAttention.__call__`, why do we add `-jnp.inf` to masked
   logits before the softmax instead of, say, zeroing them out after?
3. `generate()` always pads the context to a fixed `block_size` before
   calling the (jit'd) forward pass. What would happen to compile time if
   it instead passed in the exact, growing length of the sequence so far?
