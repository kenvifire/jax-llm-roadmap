# JAX LLM Roadmap

A 12-week, 6–8 hrs/week hands-on path to learning LLMs with JAX on TPU. Written
from the perspective of a Customer Solutions Engineer on Google Cloud TPU AI
infra, picking up the ML side of the stack.

**Rule: every phase ends with working code committed to this repo.**

## The plan

| Weeks | Phase | Goal | Deliverable |
|---|---|---|---|
| 1–2 | [01 JAX fundamentals](phases/01-jax-fundamentals/README.md) | `jit`, `grad`, `vmap`, pytrees, PRNG | MLP in pure JAX |
| 3–4 | [02 Flax + transformer](phases/02-flax-transformer/README.md) | Flax NNX, Optax, attention | Char-level transformer |
| 5–6 | [03 nanoGPT in JAX](phases/03-nanogpt-jax/README.md) | GPT-2 small (124M) on one TPU, Orbax checkpoints, Grain data loading | Training run + checkpoints |
| 7–9 | [04 Scaling on TPU](phases/04-scaling-on-tpu/README.md) | "How to Scale Your Model", `jax.sharding`, DP/FSDP/TP, XProf profiling | Sharded GPT + profile notes |
| 10–11 | [05 MaxText](phases/05-maxtext/README.md) | Gemma/Llama train + fine-tune, read sharding config and Pallas attention kernels | Fine-tune run |
| 12 | [06 Inference](phases/06-inference/README.md) | KV cache, batching, quantization, serve with JetStream or vLLM TPU | Served model + write-up |

See [RESOURCES.md](RESOURCES.md) for the reading list and [LOG.md](LOG.md) for
the weekly log.

## Progress checklist

- [ ] 01 — JAX fundamentals: MLP in pure JAX
- [ ] 02 — Flax + transformer: char-level transformer trains and samples
- [ ] 03 — nanoGPT in JAX: GPT-2 small training run + checkpoints
- [ ] 04 — Scaling on TPU: sharded GPT + profile notes
- [ ] 05 — MaxText: fine-tune run on Gemma/Llama
- [ ] 06 — Inference: served model + write-up

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All starter code through Phase 02 runs on a laptop CPU — no TPU required to
get started. From Phase 03 onward you'll want TPU access; once on a TPU VM,
also install the TPU build of JAX:

```bash
pip install -U "jax[tpu]"
```

## Compute options

- **Kaggle** — free TPU v3-8 / v5e notebooks, good for Phases 01–03.
- **Colab** — free/paid TPU runtimes, quick iteration for small experiments.
- **TPU Research Cloud (TRC)** — apply for free Cloud TPU quota for research
  and learning use: https://sites.research.google/trc/
- **Cloud TPU v5e / v6e** — on-demand or reserved Cloud TPU VMs for the
  scaling (Phase 04) and MaxText (Phase 05) work, where you need multi-chip
  slices.
