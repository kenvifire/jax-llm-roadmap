# 01 — JAX fundamentals

## Goal

Get comfortable with the core JAX transformations — `jit`, `grad`, `vmap` —
and the two ideas that make them composable: pytrees and explicit PRNG keys.

## Study

- JAX tutorials (official docs): https://docs.jax.dev/en/latest/tutorials.html
  — read through "Automatic differentiation", "Just-in-time compilation",
  and "Automatic vectorization".
- JAX pytrees: https://docs.jax.dev/en/latest/pytrees.html
- JAX PRNG design: https://docs.jax.dev/en/latest/random-numbers.html

## Files

- `mlp.py` — a 2-layer MLP in pure JAX (no Flax), trained on sklearn's
  `load_digits` dataset.

## Do

- [ ] Run `mlp.py` and confirm it trains to ~89%+ test accuracy.
- [ ] Rewrite `forward_single` to add a third layer, re-run, and compare
      accuracy.
- [ ] Replace the hand-written SGD update in `train_step` with
      `jax.tree.map` over a momentum term you add yourself (no Optax).
- [ ] Break something on purpose: remove the `jax.jit` on `train_step` and
      time 1000 steps before vs. after — note the slowdown.

## Check yourself

1. Why does `forward_single` take one example at a time, and what does
   `jax.vmap` do to turn it into a batched function? What would break if
   you passed a batch of inputs directly into `forward_single` without
   `vmap`?
2. Why do we split `key` before every call to `jax.random.split` /
   `jax.random.randint` instead of reusing the same key? What would go
   wrong if we didn't?
3. `train_step` is wrapped in `jax.jit`. What happens to `jax.jit`'s
   internal compilation cache if you call it once with a `(64, 64)` batch
   and once with a `(32, 64)` batch?
