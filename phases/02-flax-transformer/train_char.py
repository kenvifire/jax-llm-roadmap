"""Char-level language model training for the Phase 2 GPT.

Uses input.txt if present in this directory, otherwise falls back to
concatenating the repo's own .py/.md files so the script runs out of the
box with no external dataset. Trains with Optax AdamW, gradient clipping,
and a warmup+cosine learning-rate schedule. Train/eval steps are nnx.jit'd.
"""
import pathlib

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from model import GPT, GPTConfig

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

BLOCK_SIZE = 64
BATCH_SIZE = 32
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
NUM_STEPS = 300
EVAL_EVERY = 50
LEARNING_RATE = 3e-4
WARMUP_STEPS = 20


def load_text() -> str:
    input_path = HERE / "input.txt"
    if input_path.exists():
        return input_path.read_text(encoding="utf-8")

    chunks = []
    for pattern in ("*.py", "*.md"):
        for path in sorted(REPO_ROOT.rglob(pattern)):
            if ".venv" in path.parts or ".git" in path.parts:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(chunks)
    if not text:
        text = "the quick brown fox jumps over the lazy dog. " * 200
    return text


def build_batches(data, key, batch_size, block_size):
    n = data.shape[0]
    ix = jax.random.randint(key, (batch_size,), 0, n - block_size - 1)
    offsets = jnp.arange(block_size + 1)
    windows = data[ix[:, None] + offsets[None, :]]  # vectorized gather, no Python loop
    x = windows[:, :-1]
    y = windows[:, 1:]
    return x, y


def loss_fn(model, x, y):
    logits = model(x)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(y, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


@nnx.jit
def train_step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model, x, y):
    return loss_fn(model, x, y)


def main():
    text = load_text()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    data = jnp.array([stoi[ch] for ch in text], dtype=jnp.int32)
    split = int(0.9 * data.shape[0])
    train_data, val_data = data[:split], data[split:]
    print(f"dataset: {len(text)} chars, vocab_size={vocab_size}")

    config = GPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE, n_layer=N_LAYER,
        n_head=N_HEAD, n_embd=N_EMBD,
    )
    model = GPT(config, rngs=nnx.Rngs(0))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LEARNING_RATE, warmup_steps=WARMUP_STEPS,
        decay_steps=NUM_STEPS,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    key = jax.random.key(0)
    for step in range(NUM_STEPS):
        key, batch_key = jax.random.split(key)
        x, y = build_batches(train_data, batch_key, BATCH_SIZE, BLOCK_SIZE)
        loss = train_step(model, optimizer, x, y)

        if step % EVAL_EVERY == 0 or step == NUM_STEPS - 1:
            key, val_key = jax.random.split(key)
            vx, vy = build_batches(val_data, val_key, BATCH_SIZE, BLOCK_SIZE)
            val_loss = eval_step(model, vx, vy)
            print(f"step {step:5d}  train_loss {loss:.4f}  val_loss {val_loss:.4f}")

    print("\n--- sample ---")
    context = jnp.zeros((1, 1), dtype=jnp.int32)
    out = model.generate(context, max_new_tokens=200, key=jax.random.key(42))
    sample = "".join(itos[int(i)] for i in out[0])
    print(sample)


if __name__ == "__main__":
    main()
