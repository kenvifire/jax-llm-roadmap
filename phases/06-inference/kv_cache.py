"""One causal self-attention layer decoded two ways: naive (full recompute
of Q/K/V for every buffered position at each step) vs. with a preallocated
KV cache (dynamic_update_slice to write only the new token's K/V, masking
positions beyond the current length). Verifies both produce the same
output to ~1e-7 and times the speedup.

Both decode loops run as a single jax.lax.scan (one compiled call for the
whole generation) rather than a Python loop of per-step jit calls -- with
a Python loop, dispatch overhead per call swamps the real compute
difference between the two approaches and the speedup never shows up.

Note: isolated to one layer, the cached version here only avoids
recomputing Q/K/V projections, not the (equally sized, for both versions)
masked attention matmul over the fixed MAX_LEN buffer -- so the speedup
measured below is real but modest (single digits to tens of x), not the
100x+ you see in full multi-layer serving, where every layer's projections
*and* the embedding lookup are skipped on every cached step.
"""
import time

import jax
import jax.numpy as jnp
from jax import lax

D_MODEL = 512
N_HEAD = 8
HEAD_DIM = D_MODEL // N_HEAD
MAX_LEN = 512
NUM_GEN_STEPS = MAX_LEN

key = jax.random.key(0)
k_wq, k_wk, k_wv, k_wo, k_x0 = jax.random.split(key, 5)
w_q = jax.random.normal(k_wq, (D_MODEL, D_MODEL)) * 0.02
w_k = jax.random.normal(k_wk, (D_MODEL, D_MODEL)) * 0.02
w_v = jax.random.normal(k_wv, (D_MODEL, D_MODEL)) * 0.02
w_o = jax.random.normal(k_wo, (D_MODEL, D_MODEL)) * 0.02

SCALE = 1.0 / jnp.sqrt(HEAD_DIM)
POSITIONS = jnp.arange(MAX_LEN)


def split_heads(t):
    # t: (T, D_MODEL) -> (N_HEAD, T, HEAD_DIM)
    T = t.shape[0]
    return t.reshape(T, N_HEAD, HEAD_DIM).transpose(1, 0, 2)


def attend(q_t, k, v, t):
    """q_t: (N_HEAD, 1, HEAD_DIM); k, v: (N_HEAD, MAX_LEN, HEAD_DIM)."""
    mask = (POSITIONS <= t)[None, None, :]
    logits = jnp.einsum("htd,hsd->hts", q_t, k) * SCALE
    logits = jnp.where(mask, logits, -jnp.inf)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("hts,hsd->htd", weights, v)
    return out.transpose(1, 0, 2).reshape(1, D_MODEL) @ w_o


@jax.jit
def run_naive(x_full):
    """At every step, recomputes Q/K/V for the *entire* MAX_LEN buffer."""

    def step(x_buf, t):
        x_buf = lax.dynamic_update_slice(x_buf, x_full[t][None, :], (t, 0))
        q = split_heads(x_buf @ w_q)
        k = split_heads(x_buf @ w_k)
        v = split_heads(x_buf @ w_v)
        q_t = lax.dynamic_slice(q, (0, t, 0), (N_HEAD, 1, HEAD_DIM))
        out = attend(q_t, k, v, t)
        return x_buf, out[0]

    x_buf0 = jnp.zeros((MAX_LEN, D_MODEL))
    _, outs = lax.scan(step, x_buf0, jnp.arange(NUM_GEN_STEPS))
    return outs


@jax.jit
def run_cached(x_full):
    """At every step, only computes Q/K/V for the single new token."""

    def step(carry, t):
        k_cache, v_cache = carry
        x_t = x_full[t][None, :]
        q_t = split_heads(x_t @ w_q)
        k_t = split_heads(x_t @ w_k)
        v_t = split_heads(x_t @ w_v)
        k_cache = lax.dynamic_update_slice(k_cache, k_t, (0, t, 0))
        v_cache = lax.dynamic_update_slice(v_cache, v_t, (0, t, 0))
        out = attend(q_t, k_cache, v_cache, t)
        return (k_cache, v_cache), out[0]

    k0 = jnp.zeros((N_HEAD, MAX_LEN, HEAD_DIM))
    v0 = jnp.zeros((N_HEAD, MAX_LEN, HEAD_DIM))
    _, outs = lax.scan(step, (k0, v0), jnp.arange(NUM_GEN_STEPS))
    return outs


def time_fn(fn, *args, warmup=1):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    start = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def main():
    x_full = jax.random.normal(k_x0, (NUM_GEN_STEPS, D_MODEL))

    naive_out, naive_time = time_fn(run_naive, x_full)
    cached_out, cached_time = time_fn(run_cached, x_full)

    max_diff = jnp.max(jnp.abs(naive_out - cached_out))
    print(f"max abs diff between naive and cached outputs: {max_diff:.2e}")
    assert max_diff < 1e-5, "naive and cached decoding diverged"

    speedup = naive_time / cached_time
    print(f"naive decode:  {naive_time*1000:.2f} ms")
    print(f"cached decode: {cached_time*1000:.2f} ms")
    print(f"speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
