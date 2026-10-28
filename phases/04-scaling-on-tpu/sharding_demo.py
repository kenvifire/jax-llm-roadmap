"""Sharding demo: data parallelism, Megatron tensor parallelism, and
FSDP+TP weight sharding, all on 8 *fake* devices (CPU, no TPU needed).

Run as-is: this script sets XLA_FLAGS itself (before importing jax) to fake
8 host devices on CPU via --xla_force_host_platform_device_count=8, so you
don't need to export it in your shell first.

Pitfall: in recent JAX, jax.make_mesh defaults to Explicit axes, under
which a tensor-parallel matmul fails with "Contracting dimensions are
sharded". Passing axis_types=(AxisType.Auto, AxisType.Auto) below avoids
that by letting XLA's SPMD partitioner infer the collectives.
"""
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, AxisType

assert len(jax.devices()) == 8, "expected 8 fake devices; set XLA_FLAGS before importing jax"

mesh = jax.make_mesh((4, 2), ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))

BATCH, D_MODEL, D_FF = 32, 64, 256


def count_collectives(compiled_text: str) -> dict:
    counts = {}
    for op in ("all-reduce", "all-gather", "reduce-scatter"):
        counts[op] = compiled_text.count(op + "(")
    return counts


def demo_data_parallel():
    print("\n=== Data parallel activations ===")
    key = jax.random.key(0)
    x = jax.random.normal(key, (BATCH, D_MODEL))
    w = jax.random.normal(jax.random.key(1), (D_MODEL, D_MODEL))

    x_sharded = jax.device_put(x, NamedSharding(mesh, P("data", None)))
    w_replicated = jax.device_put(w, NamedSharding(mesh, P(None, None)))

    @jax.jit
    def fwd(x, w):
        return x @ w

    y_sharded = fwd(x_sharded, w_replicated)
    y_ref = x @ w

    print("x sharding:")
    jax.debug.visualize_array_sharding(x_sharded)
    print("y sharding:")
    jax.debug.visualize_array_sharding(y_sharded)

    assert jnp.allclose(y_sharded, y_ref, atol=1e-5), "DP result mismatch"
    compiled = fwd.lower(x_sharded, w_replicated).compile()
    print("collective ops:", count_collectives(compiled.as_text()))
    print("DP demo OK: batch-parallel matmul needs no collectives.")


def demo_tensor_parallel():
    print("\n=== Megatron tensor-parallel MLP ===")
    key = jax.random.key(2)
    x = jax.random.normal(key, (BATCH, D_MODEL))
    w1 = jax.random.normal(jax.random.key(3), (D_MODEL, D_FF)) * 0.02
    w2 = jax.random.normal(jax.random.key(4), (D_FF, D_MODEL)) * 0.02

    x_sharded = jax.device_put(x, NamedSharding(mesh, P("data", None)))
    w1_sharded = jax.device_put(w1, NamedSharding(mesh, P(None, "model")))  # column-split
    w2_sharded = jax.device_put(w2, NamedSharding(mesh, P("model", None)))  # row-split

    @jax.jit
    def mlp(x, w1, w2):
        h = jax.nn.gelu(x @ w1)
        return h @ w2  # contracts over the "model"-sharded F axis -> all-reduce

    y_sharded = mlp(x_sharded, w1_sharded, w2_sharded)
    y_ref = jax.nn.gelu(x @ w1) @ w2

    print("w1 sharding (column-split):")
    jax.debug.visualize_array_sharding(w1_sharded)
    print("w2 sharding (row-split):")
    jax.debug.visualize_array_sharding(w2_sharded)

    assert jnp.allclose(y_sharded, y_ref, atol=1e-4), "TP result mismatch"
    compiled = mlp.lower(x_sharded, w1_sharded, w2_sharded).compile()
    print("collective ops:", count_collectives(compiled.as_text()))
    print("TP demo OK: row/column split requires an all-reduce to sum partial sums.")


def demo_fsdp_tp_weights():
    print("\n=== FSDP + TP combined weight sharding ===")
    key = jax.random.key(5)
    x = jax.random.normal(key, (BATCH, D_MODEL))
    w1 = jax.random.normal(jax.random.key(6), (D_MODEL, D_FF)) * 0.02
    w2 = jax.random.normal(jax.random.key(7), (D_FF, D_MODEL)) * 0.02

    # x is replicated here: FSDP shards *parameters*, not activations.
    x_replicated = jax.device_put(x, NamedSharding(mesh, P(None, None)))
    # w1: FSDP shards the input (contracting) dim across "data", TP shards
    # the output dim across "model" -> each device holds a (D/4, F/2) shard.
    w1_sharded = jax.device_put(w1, NamedSharding(mesh, P("data", "model")))
    # w2: TP shards the input (contracting) dim across "model", FSDP shards
    # the output dim across "data".
    w2_sharded = jax.device_put(w2, NamedSharding(mesh, P("model", "data")))

    @jax.jit
    def mlp(x, w1, w2):
        h = jax.nn.gelu(x @ w1)
        return h @ w2

    y_sharded = mlp(x_replicated, w1_sharded, w2_sharded)
    y_ref = jax.nn.gelu(x @ w1) @ w2

    print("w1 sharding (FSDP rows + TP columns):")
    jax.debug.visualize_array_sharding(w1_sharded)
    print("w2 sharding (TP rows + FSDP columns):")
    jax.debug.visualize_array_sharding(w2_sharded)

    assert jnp.allclose(y_sharded, y_ref, atol=1e-4), "FSDP+TP result mismatch"
    compiled = mlp.lower(x_replicated, w1_sharded, w2_sharded).compile()
    print("collective ops:", count_collectives(compiled.as_text()))
    print("FSDP+TP demo OK: the SPMD partitioner inserts whatever collectives are")
    print("needed to reconstruct full contracting dims across both mesh axes --")
    print("inspect compiled.as_text() yourself to see exactly which ones it chose.")


if __name__ == "__main__":
    print(f"mesh: {mesh}")
    demo_data_parallel()
    demo_tensor_parallel()
    demo_fsdp_tp_weights()
