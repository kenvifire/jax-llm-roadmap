# 05 — MaxText

## Goal

Go from hand-rolled training code to a production training framework.
Train and fine-tune a real open-weights model (Gemma or Llama) with
MaxText, and read through its sharding config and Pallas attention kernels
to see how the Phase 04 concepts show up in production code.

## Study

- MaxText: https://github.com/AI-Hypercomputer/maxtext
- Pallas docs: https://docs.jax.dev/en/latest/pallas/index.html

**Don't follow a hardcoded install/run recipe for MaxText here** — its
entry points and config format change between releases. Always start from
MaxText's own README and docs in the repo above for the current setup and
training/fine-tuning commands.

## Files

No starter script for this phase — it's config-driven. Keep your own
config files and notes here as you go, e.g.:

- `configs/` — your MaxText config overrides for the model(s) you train.
- `NOTES.md` — what you changed, why, and what happened.

## Do

- [ ] Follow MaxText's current README to get a small training run going
      on your TPU setup.
- [ ] Fine-tune a Gemma or Llama checkpoint on a small dataset.
- [ ] Find and read the sharding config for the model you're running —
      map its axes back to the DP/FSDP/TP concepts from Phase 04.
- [ ] Find MaxText's Pallas attention kernel(s) and read through one,
      comparing it to the hand-written attention from Phase 02.
- [ ] Commit your config + a short write-up of the fine-tune run.

## Check yourself

1. How does MaxText's sharding config map onto the mesh axes and
   `PartitionSpec`s you used by hand in Phase 04?
2. What does the Pallas attention kernel do differently from the plain
   `jnp.einsum`-based attention in Phase 02's `model.py`, and why?
3. What's different about fine-tuning a pretrained checkpoint versus
   training from scratch, in terms of what you need to get right
   (learning rate, data, checkpoint loading)?
