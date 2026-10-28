# 06 — Inference

## Goal

Understand the pieces that make LLM serving fast: KV caching, batching,
and quantization — then actually serve a model with a real serving stack.

## Study

- JetStream: https://github.com/AI-Hypercomputer/JetStream
- vLLM TPU: https://github.com/vllm-project/vllm
- Kwon et al., *Efficient Memory Management for LLM Serving* (PagedAttention)
  (see [RESOURCES.md](../../RESOURCES.md))

## Files

- `kv_cache.py` — one causal self-attention layer decoded two ways: naive
  full recompute vs. a preallocated KV cache (`dynamic_update_slice` +
  masking). Verifies the two match to ~1e-7 and times the difference.

## Do

- [ ] Run `kv_cache.py` and confirm the naive and cached outputs match,
      and read the speedup number — and the note in the file about why a
      single-layer CPU benchmark understates the real-world gain.
- [ ] Wire a KV cache into the full Phase 02/03 GPT's `generate()` method
      (not just one layer) and re-measure tokens/sec.
- [ ] Add simple batching (multiple sequences decoded together) on top of
      the cached generate, and measure throughput vs. single-sequence.
- [ ] Try post-training quantization (e.g. int8 weights) on the model and
      compare output quality and decode speed.
- [ ] Install JetStream or vLLM TPU per their own docs and serve your
      fine-tuned Phase 05 model; write up latency/throughput numbers.

## Check yourself

1. In `kv_cache.py`, exactly what does the cached version skip computing
   on each step that the naive version redoes? What does it *not* skip?
2. Why does `cached_step` use `jax.lax.dynamic_update_slice` into a
   preallocated `(N_HEAD, MAX_LEN, HEAD_DIM)` buffer instead of
   concatenating the new K/V onto a growing array each step?
3. What's the difference between the kind of batching a KV cache enables
   (batching independent sequences) and the model/tensor-parallel sharding
   from Phase 04? Could you use both at once, and why would you want to?
