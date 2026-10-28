# 04 — Scaling on TPU

## Goal

Understand how training is parallelized across many TPU chips: data
parallelism, tensor parallelism, and FSDP, and how to read a profile to see
where time actually goes.

## Study

- *How to Scale Your Model* (the Scaling Book): https://jax-ml.github.io/scaling-book/
- JAX distributed arrays & automatic parallelization: https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- JAX `shard_map`: https://docs.jax.dev/en/latest/notebooks/shard_map.html
- JAX profiling (XProf) docs: https://docs.jax.dev/en/latest/profiling.html
- Shoeybi et al., *Megatron-LM*; Rajbhandari et al., *ZeRO* (see [RESOURCES.md](../../RESOURCES.md))

## Files

- `sharding_demo.py` — fakes 8 devices on CPU, builds a 4x2 `("data",
  "model")` mesh, and demos data-parallel activations, Megatron-style
  tensor parallelism (column-split then row-split MLP), and combined
  FSDP+TP weight sharding. Visualizes every sharding and checks the
  sharded result against the unsharded one, then counts collective ops
  (all-reduce / all-gather / reduce-scatter) in the compiled HLO.

## Do

- [ ] Run `sharding_demo.py` and read the `visualize_array_sharding`
      output for each case until the device layout makes sense.
- [ ] Open the compiled HLO text yourself (`compiled.as_text()`, as used
      inside `count_collectives`) and find the actual collective ops by
      name, not just the count.
- [ ] Take the Phase 03 GPT training step and shard it across this same
      mesh using `jax.jit`'s `in_shardings`/`out_shardings` or `shard_map`.
- [ ] Profile a few training steps with JAX's profiler and open the trace
      in XProf/Perfetto; write down what the biggest chunk of step time is.
- [ ] Write up your profile notes (a paragraph or two) next to the sharded
      training code.

## Check yourself

1. Why does the Megatron tensor-parallel MLP (`w1` column-split, `w2`
   row-split) need an all-reduce, but the data-parallel case needs no
   collectives at all for the matmul itself?
2. `jax.make_mesh` defaults to `AxisType.Explicit` in recent JAX, which
   makes a contracting-dimension-sharded matmul fail outright instead of
   silently inserting a collective. Why might that default exist, and
   when would you actually want `Explicit` over `Auto`?
3. If you doubled the "model" axis size and halved the "data" axis size
   (keeping total devices fixed), what would you expect to happen to the
   all-reduce traffic volume in the tensor-parallel demo?
