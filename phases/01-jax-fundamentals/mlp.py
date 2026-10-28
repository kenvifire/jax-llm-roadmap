"""2-layer MLP in pure JAX (no Flax) trained on sklearn's digits dataset.

Demonstrates: params as a pytree, vmap over a single-example forward pass,
value_and_grad, a jit'd SGD step using jax.tree.map, and explicit PRNG key
splitting.
"""
import jax
import jax.numpy as jnp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

HIDDEN_DIM = 64
NUM_CLASSES = 10
LEARNING_RATE = 0.1
NUM_STEPS = 2000
BATCH_SIZE = 64


def init_params(key, in_dim, hidden_dim, out_dim):
    key_w1, key_w2 = jax.random.split(key)
    w1 = jax.random.normal(key_w1, (in_dim, hidden_dim)) * jnp.sqrt(1.0 / in_dim)
    b1 = jnp.zeros((hidden_dim,))
    w2 = jax.random.normal(key_w2, (hidden_dim, out_dim)) * jnp.sqrt(1.0 / hidden_dim)
    b2 = jnp.zeros((out_dim,))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def forward_single(params, x):
    """Forward pass for a single example; batched with vmap."""
    h = jnp.maximum(x @ params["w1"] + params["b1"], 0.0)
    logits = h @ params["w2"] + params["b2"]
    return logits


forward_batch = jax.vmap(forward_single, in_axes=(None, 0))


def loss_fn(params, x_batch, y_batch):
    logits = forward_batch(params, x_batch)
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(y_batch, NUM_CLASSES)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


@jax.jit
def train_step(params, x_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
    params = jax.tree.map(lambda p, g: p - LEARNING_RATE * g, params, grads)
    return params, loss


def accuracy(params, x, y):
    preds = jnp.argmax(forward_batch(params, x), axis=-1)
    return jnp.mean(preds == y)


def main():
    key = jax.random.key(0)

    digits = load_digits()
    x = digits.data.astype("float32")
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)
    y = digits.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    x_train = jnp.array(x_train)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test)
    y_test = jnp.array(y_test)

    key, init_key = jax.random.split(key)
    params = init_params(init_key, x_train.shape[1], HIDDEN_DIM, NUM_CLASSES)

    num_train = x_train.shape[0]
    for step in range(NUM_STEPS):
        key, batch_key = jax.random.split(key)
        idx = jax.random.randint(batch_key, (BATCH_SIZE,), 0, num_train)
        params, loss = train_step(params, x_train[idx], y_train[idx])
        if step % 500 == 0 or step == NUM_STEPS - 1:
            test_acc = accuracy(params, x_test, y_test)
            print(f"step {step:5d}  loss {loss:.4f}  test_acc {test_acc:.4f}")

    final_acc = accuracy(params, x_test, y_test)
    print(f"final test accuracy: {final_acc:.4f}")
    assert final_acc > 0.85, "expected roughly 89% test accuracy"


if __name__ == "__main__":
    main()
