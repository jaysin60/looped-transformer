"""Generate the Colab notebook for Track B (looped upcycle)."""

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


cells = [
    md("""
# Looped Transformer — Track B (upcycle Qwen2.5-Coder)

Train a **looped (latent-recurrent)** version of a small pretrained coding model on an
open-source **agentic + coding** mix, then measure whether deeper loops (`k=1,2,3,4`)
improve held-out loss.

**What this notebook does**
1. checks the GPU and installs dependencies
2. mounts Drive and logs in to W&B
3. builds (or copies) the packed agentic+coding dataset
4. upcycles `Qwen/Qwen2.5-Coder-1.5B` into a looped model with LoRA
5. trains with loop-count sampling + truncated backprop (TBPTL depth 1)
6. evaluates the k-sweep

**Key design:** the first loop is the untouched pretrained stack (so `k=1` == base model).
Later loops are a **gated residual refinement** initialized at gate 0, so `k>1` starts
identical to `k=1` and learns how much the recurrent block should contribute.
"""),
    md("## 1. GPU + dependencies"),
    code("""
!nvidia-smi
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
"""),
    code("""
# Colab already ships torch; install the rest quietly.
!pip install -q -U transformers datasets accelerate safetensors wandb
"""),
    md("## 2. Drive + W&B"),
    code("""
import os
from google.colab import drive, userdata

drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/looped-transformer'
os.makedirs(DRIVE, exist_ok=True)

import wandb
wandb.login(key=userdata.get('WANDB_API_KEY'))
print('Drive:', DRIVE)
"""),
    md("""
## 3. Get the code

Either clone the repo, or upload the `looped-transformer` folder to Drive and copy it.
Set `REPO_URL` if you have a GitHub repo.
"""),
    code("""
REPO_URL = ''  # e.g. 'https://github.com/you/looped-transformer.git'
CODE_DIR = '/content/looped-transformer'

import os
if REPO_URL:
    !git clone $REPO_URL $CODE_DIR
else:
    # fallback: copy a pre-uploaded code folder from Drive
    !cp -r "$DRIVE/code" $CODE_DIR

%cd $CODE_DIR
!ls
"""),
    md("""
## 4. Build the packed dataset

The mix is OpenCoder (coding), OpenThoughts-Agent (agentic), and R2E-Gym (SWE).
Assistant-only labels are produced with a ChatML `{% generation %}` template.
For a quick run, cap the examples; for a longer run, raise them.
"""),
    code("""
%cd /content/looped-transformer

# Option A: build on Colab (downloads ~2-3 GB, tokenizes ~9K examples)
!PYTHONPATH=. python data/build_mix.py build \
  --out data/packed \
  --tokenizer Qwen/Qwen2.5-Coder-1.5B \
  --seq-len 2048 \
  --max-examples 3000 \
  --max-sequences 8000

# Option B: copy a prebuilt pack from Drive (faster; upload data/packed_medium first)
# !cp -r "$DRIVE/packed_medium" data/packed
"""),
    md("""
## 5. Train (L4 / A100 path)

- `Qwen/Qwen2.5-Coder-1.5B`, 28 layers → input 6 / recurrent 16 / output 6
- LoRA rank 32 on all blocks; loop norms + gate trainable
- `--no-inject-input`: refine the state (`u=s`) rather than adding the input every loop
- gate LR ×100 so the scalar gate can move
- `max_steps` counts micro-batches; `grad_accum=8` → ~1000 optimizer steps at 2048 tokens

If you hit OOM, switch to `Qwen/Qwen2.5-Coder-0.5B` and the T4 cell below.
"""),
    code("""
%cd /content/looped-transformer
!PYTHONPATH=. python -m looped.trainer \
  --base-model Qwen/Qwen2.5-Coder-1.5B \
  --data-dir data/packed \
  --out-dir "$DRIVE/runs/l4_gated" \
  --n-input 6 --n-recurrent 16 --n-output 6 \
  --seq-len 2048 --batch-size 1 --grad-accum 8 \
  --max-steps 8000 --eval-every 500 --eval-batches 16 \
  --save-every 1000 --device cuda --dtype bfloat16 \
  --lora-rank 32 --learning-rate 1e-4 --warmup-steps 100 \
  --no-inject-input --gate-lr-mult 100 --wandb
"""),
    md("""
### Optional: non-looped baseline

Same base model, same data, same LoRA, but trained with **`k=1` only**
(`--max-k 1 --override-frac 0`). Compare its held-out loss to the looped model's
`k=1` and `k=2/4`. This roughly doubles the compute; run only if you have budget.
"""),
    code("""
# %cd /content/looped-transformer
# !PYTHONPATH=. python -m looped.trainer \
#   --base-model Qwen/Qwen2.5-Coder-1.5B \
#   --data-dir data/packed \
#   --out-dir "$DRIVE/runs/l4_baseline" \
#   --n-input 6 --n-recurrent 16 --n-output 6 \
#   --seq-len 2048 --batch-size 1 --grad-accum 8 \
#   --max-steps 4000 --eval-every 500 --eval-batches 16 \
#   --save-every 1000 --device cuda --dtype bfloat16 \
#   --lora-rank 32 --learning-rate 1e-4 --warmup-steps 100 \
#   --max-k 1 --override-frac 0 --no-inject-input --gate-lr-mult 100 --wandb
"""),
    md("## 6. T4 fast path (0.5B, 1024 tokens)"),
    code("""
%cd /content/looped-transformer
!PYTHONPATH=. python data/build_mix.py build \
  --out data/packed_t4 \
  --tokenizer Qwen/Qwen2.5-Coder-0.5B \
  --seq-len 1024 \
  --max-examples 3000 \
  --max-sequences 8000
!PYTHONPATH=. python -m looped.trainer \
  --base-model Qwen/Qwen2.5-Coder-0.5B \
  --data-dir data/packed_t4 \
  --out-dir "$DRIVE/runs/t4_gated" \
  --n-input 5 --n-recurrent 14 --n-output 5 \
  --seq-len 1024 --batch-size 2 --grad-accum 8 \
  --max-steps 6000 --eval-every 500 --eval-batches 16 \
  --save-every 1000 --device cuda --dtype float32 \
  --lora-rank 16 --learning-rate 1e-4 --warmup-steps 100 \
  --no-inject-input --gate-lr-mult 100 --wandb
"""),
    md("""
## 7. Evaluate the k-sweep

Load the final adapter and compare held-out loss at `k=1,2,3,4` (and `k=6` for
extrapolation). The interesting result is whether `k=2` (or `k=4`) beats `k=1`.
"""),
    code("""
import numpy as np, torch
from looped.model import LoopedCausalLM
from looped.trainer import load_packed, evaluate

device = torch.device('cuda')
ids, labels, meta = load_packed('/content/looped-transformer/data/packed')

# same random split as the trainer (seed=12181531)
rng = np.random.default_rng(12181531)
perm = rng.permutation(len(ids))
n_val = min(max(20, int(0.05 * len(ids))), 500)
val_ids, val_labels = ids[perm[:n_val]], labels[perm[:n_val]]

model = LoopedCausalLM.load_adapter('/content/drive/MyDrive/looped-transformer/runs/l4_gated/adapter_final',
                                    dtype='bfloat16').to(device)
model.eval()

for k in (1, 2, 3, 4, 6):
    loss = evaluate(model, val_ids, val_labels, n_loops=k, batch_size=1,
                    max_batches=32, device=device, tbptt_depth=0)
    print(f'k={k} val_loss={loss:.4f}')
print('gate:', model.loop_gate.item() if model.loop_gate is not None else None)
"""),
    md("""
## 8. What to look for / next steps

- **k=2 ≥ k=1**: the recurrent block is contributing useful refinement.
- **gate ≈ 0**: the loop is being ignored; try more steps, a higher gate LR, or `--inject-input`.
- **k=4 > k=2**: deeper loops help; check whether the data is hard enough (agentic trajectories).
- Next ablations: `--inject-input`, `--normalize-loop-output`, TBPTT depth 0 vs 1,
  and a non-looped baseline with the same LoRA budget.
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path("notebooks/01_colab_train_l4.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(cells)} cells")
