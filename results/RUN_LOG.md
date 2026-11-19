# Run log

## 2026-11-16 — local M6 smoke tests

**Environment:** MacBook Pro M6, 24 GB unified memory, Python 3.13.18, torch 2.15.0 (MPS),
transformers 5.23.0. No CUDA locally; Colab T4/L4 reserved for the real run.

### Model wrapper validation

- `looped/model.py` wraps a Qwen2/Llama causal LM, partitions its decoder layers into
  input / recurrent / output blocks, and applies the recurrent block `k` times.
- **k=1 is numerically identical to the pretrained base model** when
  `first_loop_passthrough=True` (LoRA B initialized to zero):
  - base loss `14.15103436`
  - looped k=1 `14.15103340` (diff `9.5e-7`)
- With `loop_residual=True` and `loop_gate_init=0`, k=2/4/6 also equal k=1 at init,
  so the loop starts as a no-op and can only help once the gate moves.

### 0.5B smoke runs (256K-token packed slice, seq 512)

| Variant | k=1 | k=2 | k=3 | k=4 | Observation |
|---|---|---|---|---|---|
| exact Dory recurrence (norm in/out) | 0.3116 | 0.6067 | 0.6606 | 0.7471 | k=1 best; loop is out-of-distribution |
| passthrough + no output norm | 0.3100 | 0.4303 | 0.6155 | 0.6654 | k=1 still best |
| gated residual, inject input, gate LR×10 | 0.3324 | 0.3152 | 0.3026 | 0.3412 | noisy; k=3 best at step 490 |
| gated residual, **no input injection**, gate LR×100 | 0.2970 | 0.2967 | 0.2964 | 0.2964 | small monotonic gain with k |

Takeaways:
- Exact-Dory recurrence is a poor upcycling initialization for a pretrained model; the
  model needs to adapt to the normalized/re-injected state.
- The **base-preserving** recipe (first loop = base, later loops = gated residual,
  no input injection) is stable and gives a small but correctly ordered k-sweep.
- The gate learns slowly; a ×100 LR on the scalar gate helps. The effect is still tiny
  after 500 steps / 512K tokens, so the real run needs more data and steps.
- Validation sets were tiny (10–25 sequences) in these smoke runs; the real run uses a
  deterministic 5% held-out split.

### Medium packed dataset

`data/packed_medium` (built from OpenCoder + OpenThoughts-Agent + R2E-Gym):

- 6,799 examples → 4,000 packed sequences × 2,048 tokens = **8.19M tokens**
- **0 empty-label sequences** (after filtering), assistant-token fraction **36.4%**
- per-sequence assistant fraction: median 36.7%
- Important fix: long agentic system/tool messages were crowding out assistant turns
  under right-truncation. Messages are now capped at 1,500 chars and the builder
  filters examples with no assistant tokens.

### Adapter round-trip

`save_adapter` now stores the LoRA config alongside the loop config; `load_adapter`
recreates the LoRA modules before loading. Verified: gate value and 4.4M trainable
parameters survive a save/load round-trip.

### 200-step medium run (0.5B, seq 2048)

`runs/medium_200` — gated residual, no input injection, gate init 0, gate LR ×100,
LR 2e-4, random 5% held-out split.

In-training eval at step 100:

| k | val loss |
|---|---|
| 1 | 1.1563 |
| 2 | 1.1556 |
| 3 | 1.1549 |
| 4 | 1.1548 |

Final adapter (`scripts/eval_adapter.py`, random held-out, 4 batches):

| k | val loss |
|---|---|
| 1 | 1.0689 |
| 2 | 1.0685 |
| 4 | 1.0696 |

The gate oscillates around ±0.01 and ends near +0.004. Interpretation: **k=2 gives a
tiny gain; deeper loops do not help at this budget.** The loop is contributing very
little because the gate is near zero and only 200 steps / ~1.6M tokens were trained.
This is a pipeline validation, not a verdict on the architecture — the real test is
the 1.5B L4 run with ~16M tokens.

### Gate-init ablation (120 steps, 0.5B)

`runs/medium_gate05` — `--loop-gate-init 0.05`, no input injection, gate LR ×10.

- gate decreases monotonically: **0.050 → 0.019** over 120 steps.
- step-60 eval: k=1 **1.3193**, k=2 1.3218, k=3 1.3245, k=4 1.3310.
- Interpretation: a nonzero gate makes the loop **actively harmful** at this scale; the
  optimizer learns to suppress it. Combined with the gate-init-0 run (k=2 tiny gain),
  the loop refinement is not yet useful with 200 steps / ~1.6M tokens of LoRA adaptation.

### Smaller recurrent block ablation (6/10/8, 120 steps)

`runs/medium_smallrec` — more unique input/output layers, fewer shared layers.

- step-60 eval: k=1 1.3167, k=2 **1.3159**, k=3 1.3176, k=4 1.3179.
- final adapter: k=1 1.0972, k=2 1.0967, k=3 **1.0965**, k=4 1.0976.
- Same qualitative result as 5/14/5: a tiny k=2/3 gain, no deeper gain. The partition
  size is not the bottleneck at this budget.

### What this means

- The base-preserving gated residual works mechanically (k=1 == base; gate can move both ways).
- The loop needs either **much more training** (the paper used 1.5T looped tokens), a
  **less constrained adaptation** (full fine-tune or higher-rank LoRA), or a **harder,
  more reasoning-heavy evaluation** before it can show a real gain.
- The 1.5B / ~16M-token L4 run is the next meaningful step; a negative result there
  would still be a clean, reportable finding.

### Next

- Run the 1.5B L4 notebook (gate init 0, no injection) for ~16M tokens.
- Ablations worth trying if time: smaller recurrent block (e.g. 6/10/8), `--inject-input`,
  higher LoRA rank, agentic-only validation split.
- Compare against a non-looped LoRA baseline with the same budget.
