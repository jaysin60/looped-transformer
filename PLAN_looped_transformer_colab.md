# Looped Transformer on Colab — Plan v2

**Goal:** upcycle a small open pretrained LM into a looped (latent-recurrent) model, train it on an open-source **agentic + coding** dataset, and measure whether deeper loops help — reproducing the core of the Dory1.0-4B paper at a scale that fits a **few hours** of Colab time.

**Decisions locked in (2026-11-16):**
- Dataset: **open-source agentic/coding data** (no synthetic generation).
- Track: **B — upcycle a small pretrained model** into a looped model.
- Hardware: **develop/smoke-test on the local Mac (M6, 24 GB)**, then run the real experiment on **Colab T4/L4**.
- Budget: **a few hours total** of Colab time.

---

## 1. What we are reproducing (from the Dory paper)

- **Looped architecture:** input block → shared recurrent block applied `k` times → output block; the input representation `x` is re-injected every pass: `s0 = x`, `u_t = N(s_{t-1} + x)`, `s_t = N(F(u_t))`.
- **Elastic depth:** one checkpoint serves any `k`; accuracy generally rises with `k`.
- **Training:** sample `k ∈ {1,2,3,4}` per step (uniform) plus a 10% override to `k=1`; **TBPTL depth 1** (only the last loop is differentiated); loss only at the final loop.
- **Difficulty-aware loop curriculum** (paper §4.2): route harder prompts to deeper loops, `s_i = α·q_i + (1−α)·u_i`, `α=0.3`.
- **Post-training:** SFT on packed sequences, assistant-only loss, then RLVR. We will do the SFT stage (loops + curriculum); RLVR is out of scope for a few hours.

Dory itself is a normalized-GPT (nGPT). For Track B we keep the pretrained model’s normalization (RMSNorm) and apply the **loop recurrence and training recipe** on top; we can add nGPT-style normalization later as an ablation.

---

## 2. Hardware reality

| Device | VRAM | Precision | Role |
|---|---|---|---|
| Mac M6 (local) | 24 GB unified | FP32/BF16 via MPS | development, smoke tests, k-sweep sanity |
| Colab T4 | 16 GB | FP16 + GradScaler (no BF16) | fast real run with a 0.5B model |
| Colab L4 | 24 GB | BF16 | main real run with a 1.5B model |

Local `uv` 0.13.8 and Python 3.13 are available; PyTorch is **not installed yet** (first step).

---

## 3. Dataset — open-source agentic + coding mix

No synthetic data. We build a **unique mix + processing pipeline** from public datasets. The novelty is the blend, the difficulty metadata, and the loop curriculum — not the raw corpora.

### 3.1 Candidate datasets

| Dataset | Subset | Rows | License | Use |
|---|---|---|---|---|
| `OpenCoder-LLM/opc-sft-stage2` | `educational_instruct` | 118K | MIT | coding instructions + code + tests |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | default | 75K | MIT | coding instructions |
| `ise-uiuc/Magicoder-Evol-Instruct-110K` | default | 111K | MIT | harder coding instructions |
| `open-thoughts/OpenThoughts-Agent-SFT-100K` | default | 100K | Apache-2.0 (verify) | agent trajectories (SWE-Smith, SuperUser, Tezos, IssueTasks) |
| `nvidia/Nemotron-SFT-Agentic-v2` | `tool_calling` | 707K | CC BY 4.0 | single/multi-turn tool use |
| `nvidia/Nemotron-SFT-Agentic-v2` | `customer_service_838_domains` | 279K | CC BY 4.0 | multi-turn conversational agents |
| `nvidia/Nemotron-SFT-SWE-v3` | default | 238K | CC BY 4.0 | SWE agent trajectories |
| `R2E-Gym/R2EGym-SFT-Trajectories` | default | 3.2K | Apache-2.0 (verify) | SWE trajectories |
| `SWE-Gym/OpenHands-SFT-Trajectories` | default | 1.85K | Apache-2.0 (verify) | SWE trajectories |
| `bunnybhaiya/agentgym-sft-trajectories` | default | 102K | MIT | 5 agent environments (AlfWorld, BabyAI, WebShop, SciWorld, TextCraft) |

All are loadable with `datasets.load_dataset`. Prefer permissive licenses (MIT / Apache-2.0 / CC BY 4.0) and record the exact revision used.

### 3.2 Proposed mix (target ~20–30M tokens)

| Slice | Samples | Share |
|---|---|---|
| `opc-sft-stage2` educational_instruct | 20K | 25% |
| `Magicoder-OSS-Instruct-75K` | 10K | 12% |
| `OpenThoughts-Agent-SFT-100K` | 30K | 25% |
| `Nemotron-SFT-Agentic-v2` tool_calling | 12K | 18% |
| `Nemotron-SFT-SWE-v3` + `R2E-Gym` | 12K | 20% |
| **Total** | **~84K** | 100% |

Filters and processing:
- Render everything into one chat schema (ChatML / Qwen template), with `messages` + `tools` where applicable.
- **Loss mask:** train on assistant tokens only; include assistant tool-call tokens; mask system/user/tool tokens (the paper masks system/user/tool and trains reasoning + tool calls).
- **Difficulty metadata** per example: token length, number of assistant turns, tool-call count, whether it is a multi-step SWE task. Used later for loop routing.
- **Length filter:** keep examples ≤ 2048 tokens (T4: ≤ 1024); truncate or drop the rest. Pack with EOS to the sequence length to avoid padding waste.
- **Splits:** carve a held-out 1–2% from each source (disjoint seeds) for validation; keep a small task-accurate test set for exact-match evaluation.
- **Thinking mix:** optionally render some assistant turns as `<think>…</think>` reasoning and some as direct answers, matching the paper’s mixed thinking/non-thinking SFT.

Deliverable: `data/build_mix.py` → packed token shards (`.bin`/`.npy` or HF dataset) + a manifest with source/revision/license/difficulty.

---

## 4. Model — upcycle a small pretrained LM

### 4.1 Candidate base models

| Model | Layers | hidden | heads/KV | vocab | Notes |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-0.5B` | 24 | 896 | 14/2 | 151,936 | fast T4 run, coding-focused |
| `Qwen/Qwen2.5-Coder-1.5B` ⭐ | 28 | 1536 | 12/2 | 151,936 | main L4 run, coding-focused |
| `Qwen/Qwen2.5-Coder-3B` | 32? | 2048? | 16/2? | 151,936 | stretch |
| `HuggingFaceTB/SmolLM2-1.7B` | 24 | 2048 | 32/32 | 49,152 | general, small vocab, easy local test |
| `Qwen/Qwen3-1.7B` | 28? | 2048? | 16/8? | 151,936 | newer, qk-norm (more moving parts) |

**Recommendation:** `Qwen/Qwen2.5-Coder-1.5B` for the main run (coding/agentic, standard Llama-like architecture, tied embeddings, 28 layers). Use `Qwen2.5-Coder-0.5B` for the T4 run and `SmolLM2-1.7B` for local smoke tests.

### 4.2 Loop partition (Dory ratio ≈ 22% / 56% / 22%)

For 28 layers: **input 6 / recurrent 16 / output 6**. Effective depth = `28 + 16·(k−1)` → 28, 44, 60, 76 layers at k=1..4.
For 24 layers: **input 5 / recurrent 14 / output 5**.
For 32 layers: **input 7 / recurrent 18 / output 7**.

The recurrent block is the **same module object** applied `k` times (weights shared by construction), initialized from the pretrained middle layers. Input and output blocks keep their original pretrained weights.

### 4.3 Training setup

- **LoRA** on the recurrent block (and optionally input/output): `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`; rank 16–32, α=rank, dropout 0. Freeze embeddings and LM head (tied anyway).
- **Optimizer:** AdamW, β=(0.9, 0.95), no weight decay, grad clip 1.0, LR 1e-4–2e-4 with cosine decay and ~100 warmup steps.
- **Loop sampling:** per step, deterministic seeded draw from {1,2,3,4}; 10% of steps forced to k=1.
- **TBPTT depth 1:** loops `1..k−1` under `no_grad`; loop `k` with grad. Loss only at the final loop.
- **Precision:** BF16 on L4; FP16 + GradScaler on T4. Keep loss in FP32.
- **Memory:** gradient checkpointing on the recurrent block; batch 1–2 × seq 2048, grad accumulation to ~16–32k tokens/step.
- **Checkpointing:** save LoRA adapter + config to Drive every 100–200 steps; resume support.

### 4.4 First local Mac smoke test

- Create a venv with PyTorch (MPS), `transformers`, `datasets`, `peft`, `accelerate`, `safetensors`.
- Use `SmolLM2-1.7B` (small vocab, fast) or `Qwen2.5-Coder-0.5B`.
- 100–300 steps, seq 512, ~1k examples from `opc-sft-stage2`.
- Verify: loss decreases; forward/backward works for k=1..4; k-sweep eval runs; memory fits in 24 GB.
- Time: ~15–30 min.

---

## 5. Implementation modules to leverage

| Module | Use |
|---|---|
| `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`) | base model, chat templates, fast tokenizer |
| `datasets` | streaming/load the open datasets, packing |
| `peft` (`LoraConfig`, `get_peft_model`) | LoRA; if it rejects the custom looped class, use a manual `LoRALinear` |
| `accelerate` | simple single-GPU training loop, checkpointing |
| `safetensors` | save/load adapters |
| `bitsandbytes` | 4-bit QLoRA on Colab if VRAM is tight (not on Mac) |
| `wandb` | tracking (the existing notebook already uses it) |
| `lm-evaluation-harness` or a small local eval | held-out loss + HumanEval/MBPP pass@1 |
| `mrfakename/nemotron-dory-4b` Space `dory.py` | reference for the exact Dory loop recurrence (if we later add nGPT normalization) |
| `kschwethelm/looped-lm-scaling` | reference for TBPTT, injection modes, KV-budget ablations |
| `seal-rg/recurrent-pretraining` (Huginn) | reference for upcycling / prelude-recur-coda partitioning |

The existing `Copy_of_Qwopus3_5_27b_Colab.ipynb` is reused for its **dataset mixing, chat templating, assistant-only masking, W&B setup, and Drive checkpointing** patterns. Unsloth/TRL cannot train a custom looped model, so we write a small custom trainer.

---

## 6. Code structure (proposed)

```
looped-transformer/
├── PLAN_looped_transformer_colab.md      # this file
├── data/
│   ├── build_mix.py                      # download/sample/render/pack the agentic+coding mix
│   └── manifest.json                     # sources, revisions, licenses, token counts
├── looped/
│   ├── model.py                          # LoopedCausalLM wrapper (partition + recurrence)
│   ├── lora.py                           # optional manual LoRA if peft can't wrap it
│   ├── trainer.py                        # loop sampling, TBPTT, loss, checkpointing
│   └── evaluate.py                       # k-sweep, held-out loss, exact-match eval
├── notebooks/
│   ├── 00_local_smoke_mac.ipynb          # Mac MPS smoke test
│   └── 01_colab_train_l4.ipynb           # Colab training + eval
└── results/
    └── RUN_LOG.md
```

Key implementation detail: a custom `LoopedCausalLM` that reuses the base model’s embeddings, rotary, norms, and LM head, but replaces `model.layers` with `[input_layers] + [recurrent_layers] + [output_layers]` and implements the Dory forward. LoRA is applied to the `nn.Linear` projections before partitioning. TBPTT is implemented by detaching the state between early loops and only backpropagating through the final loop.

---

## 7. Experiment plan (fits a few hours)

### Run 0 — local Mac smoke (30 min)
- `SmolLM2-1.7B` or `Qwen2.5-Coder-0.5B`, 200 steps, seq 512, 1k coding examples.
- Success = loss drops, k=1..4 all run, memory < 20 GB.

### Run 1 — Colab T4, 0.5B (1–1.5 h)
- `Qwen/Qwen2.5-Coder-0.5B`, LoRA r=16, seq 1024, ~10–15M tokens, ~1000 steps.
- k sampling + TBPTT depth 1.
- Eval: held-out loss at k=1,2,3,4 + 200-example exact-match coding subset.

### Run 2 — Colab L4, 1.5B (2–3 h) ⭐ main result
- `Qwen/Qwen2.5-Coder-1.5B`, LoRA r=32, seq 2048, ~20–30M tokens, ~1500–2500 steps.
- Same recipe; add difficulty-aware loop routing (α=0.3) as a second arm if time permits.
- Eval: held-out loss k-sweep, HumanEval pass@1 (n=16) or a held-out execution set, and a non-looped baseline (same LoRA budget, k=1 only) for comparison.

### Optional Run 3 — ablations (only if time)
- TBPTT depth 0 vs 1; injection modes (additive vs concat vs passthrough); with/without RMSNorm at loop boundaries; LoRA on recurrent only vs all blocks.

---

## 8. Evaluation and success criteria

- **Primary:** held-out loss (and bits-per-byte) as a function of `k` for the same checkpoint — expect improvement from k=1 to k=2/4, matching the paper’s Figure 1.
- **Secondary:** task accuracy / exact match on held-out coding and agentic examples vs `k`.
- **Comparison:** looped model vs non-looped baseline with the same parameter/LoRA budget and data.
- **Diagnostics:** loop-state convergence (cosine distance between `s_{t-1}` and `s_t`), gradient norms, VRAM, tokens/s.
- **Success for a few hours:** a clear k-sweep with k=2 ≥ k=1, and a working training/eval pipeline that can be scaled later.

---

## 9. Risks & gotchas

- **Structural upcycling is lossy:** sharing 16 middle layers removes unique-layer capacity. Expect k=1 to be worse than the untouched base; the bet is that k≥2 recovers and exceeds it.
- **TBPTL depth 1:** if you backprop through all loops, memory and time explode; keep depth 1 for the first runs.
- **Loop collapse:** training only at k=1 makes deeper loops unstable; always sample mixed k with the 10% k=1 override.
- **MPS limitations:** no bitsandbytes; SDPA may fall back to math; use small seq/batch and FP32/BF16. It is for smoke tests only.
- **T4 has no BF16:** use FP16 + GradScaler.
- **Dataset licenses:** pin revisions and record licenses; prefer MIT/Apache/CC-BY.
- **Long agentic trajectories:** filter/truncate to the sequence length; packing matters.
- **Chat template drift:** use the base model’s own tokenizer chat template for consistency.
- **Colab session limits:** checkpoint to Drive frequently; keep runs < 3 h.

---

## 10. Implementation status (2026-11-16)

Done:
- Local env: `.venv-looped` (torch 2.15.0 MPS, transformers 5.23.0).
- `looped/model.py`: Qwen2/Llama upcycling wrapper with input/recurrent/output partition,
  TBPTL depth 1, LoRA, gradient checkpointing, adapter save/load.
- **k=1 equivalence verified**: the looped model at k=1 matches the pretrained base
  loss to `1e-6`.
- `looped/trainer.py`: deterministic loop sampling (`{1..4}` + 10% k=1 override),
  assistant-only loss, cosine LR, deterministic k-sweep, checkpointing.
- `data/build_mix.py`: OpenCoder + OpenThoughts-Agent + R2E-Gym, ChatML
  `{% generation %}` template for assistant masks, packed `.npy` output.
- `notebooks/01_colab_train_l4.ipynb`: end-to-end Colab notebook.
- Local M6 smoke tests: exact-Dory recurrence is a poor upcycling init; the
  **base-preserving gated residual** is stable and gives a small monotonic k-sweep.
  See `results/RUN_LOG.md`.

Revised upcycling recipe (implemented):
```
x = input_block(embed(tokens))
s = recurrent_block(x)            # first loop == pretrained stack
for t in 2..k:
    d = recurrent_block(norm(s))  # no input injection by default
    s = s + gate * d              # gate init 0, LR x100
h = output_block(s)
```
This makes k=1 exactly the base model and lets the loop start as a no-op, then learn.

Local result so far (0.5B, 200 steps, ~1.6M tokens, `runs/medium_200`):
- k=1 1.0689, k=2 1.0685, k=4 1.0696 on a random held-out split.
- k=2 gives a tiny gain; deeper loops do not help at this budget. The gate stays near
  zero (~0.004), so the loop contributes little. This validates the pipeline; it is not
  a verdict on the architecture. The 1.5B / 16M-token L4 run is the real test.
- Gate-init 0.05 ablation (`runs/medium_gate05`): the gate falls 0.050 → 0.019 and
  k=1 is best (1.3193 vs 1.3218/1.3245/1.3310). When forced to contribute at this
  scale, the loop is actively harmful and the optimizer suppresses it.
- Smaller recurrent block (6/10/8, `runs/medium_smallrec`) gives the same picture:
  k=2/3 tiny gain (~1e-3), no deeper gain. Partition size is not the bottleneck.

Remaining:
1. Test `--loop-gate-init 0.05` to give the recurrent block gradients immediately.
2. Run `Qwen2.5-Coder-1.5B` on Colab L4 (~16M tokens, ~1–2 h).
3. Evaluate k=1/2/3/4 on the random held-out split; log the gate value.
4. Optional ablations: `--inject-input`, `--normalize-loop-output`, TBPTL depth 0 vs 1,
   non-looped baseline with the same LoRA budget.
5. Update `results/RUN_LOG.md` with the L4 result.
