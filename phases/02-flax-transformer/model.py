"""A small GPT in Flax NNX with hand-written causal self-attention.

No nnx.MultiHeadAttention — attention is written out by hand so the QKV
projections, scaling, causal mask, and softmax are all visible. Uses
pre-norm transformer blocks, learned position embeddings, and weight tying
between the token embedding and the output projection (via wte.attend).
"""
import dataclasses

import jax
import jax.numpy as jnp
import flax.nnx as nnx


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nnx.Linear(config.n_embd, 3 * config.n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(config.n_embd, config.n_embd, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        def split_heads(t):
            return t.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale

        causal_mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        attn_logits = jnp.where(causal_mask, attn_logits, -jnp.inf)

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        out = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.c_proj(out)


class MLP(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, rngs=rngs)

    def __call__(self, x):
        x = self.c_fc(x)
        x = jax.nn.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.ln_1 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(config.n_embd, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nnx.Module):
    def __init__(self, config: GPTConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.wte = nnx.Embed(config.vocab_size, config.n_embd, rngs=rngs)
        self.wpe = nnx.Embed(config.block_size, config.n_embd, rngs=rngs)
        self.blocks = nnx.List([Block(config, rngs=rngs) for _ in range(config.n_layer)])
        self.ln_f = nnx.LayerNorm(config.n_embd, rngs=rngs)

    def __call__(self, idx):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"
        pos = jnp.arange(T)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.wte.attend(x)  # weight tying: reuse wte as the output head
        return logits

    def generate(self, idx, max_new_tokens, key):
        # Context is padded to a fixed block_size before every forward pass so
        # the jit'd call always sees the same shape and never recompiles.
        block_size = self.config.block_size

        @nnx.jit
        def forward(model, idx_cond):
            return model(idx_cond)

        for _ in range(max_new_tokens):
            T = idx.shape[1]
            if T >= block_size:
                idx_cond = idx[:, -block_size:]
                pad_len = 0
            else:
                pad_len = block_size - T
                idx_cond = jnp.pad(idx, ((0, 0), (0, pad_len)))
            logits = forward(self, idx_cond)
            last_pos = (block_size - 1) if pad_len == 0 else (T - 1)
            next_logits = logits[:, last_pos, :]
            key, sample_key = jax.random.split(key)
            next_token = jax.random.categorical(sample_key, next_logits, axis=-1)
            idx = jnp.concatenate([idx, next_token[:, None]], axis=1)
        return idx


if __name__ == "__main__":
    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config, rngs=nnx.Rngs(0))
    x = jnp.zeros((2, 16), dtype=jnp.int32)
    logits = model(x)
    assert logits.shape == (2, 16, config.vocab_size)
    print("forward pass ok, logits shape:", logits.shape)

    out = model.generate(jnp.zeros((1, 1), dtype=jnp.int32), max_new_tokens=10, key=jax.random.key(0))
    print("generated sequence:", out)
