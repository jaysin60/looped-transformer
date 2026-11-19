"""Training loop for the looped upcycled model.

Features:
  - deterministic per-step loop-count sampling from {1..max_k} with a k=1 override
  - TBPTL (only the last `tbptt_depth` loops are differentiated)
  - assistant-only labels (already masked in the packed data)
  - cosine LR with warmup, grad clipping
  - k-sweep evaluation and adapter checkpointing
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from looped.model import LoopedCausalLM, apply_lora


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B"
    data_dir: str = "data/packed"
    out_dir: str = "runs/looped"
    n_input: int = 6
    n_recurrent: int = 16
    n_output: int = 6
    loop_norm: str = "rms"
    first_loop_passthrough: bool = True
    normalize_loop_output: bool = False
    loop_residual: bool = True
    loop_gate_init: float = 0.0
    inject_input: bool = True
    gate_lr_mult: float = 10.0
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_blocks: tuple[str, ...] = ("input", "recurrent", "output")
    seq_len: int = 2048
    batch_size: int = 2
    grad_accum: int = 8
    max_k: int = 4
    override_frac: float = 0.1
    tbptt_depth: int = 1
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    max_steps: int = 1000
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    clip_grad: float = 1.0
    eval_every: int = 100
    eval_batches: int = 8
    save_every: int = 200
    seed: int = 12181531
    device: str = "cuda"
    dtype: str = "bfloat16"
    grad_ckpt: bool = True
    wandb: bool = False
    seed_step_offset: int = 0


def sample_loop_count(step: int, max_k: int, override_frac: float, seed: int) -> int:
    """Deterministic per-step loop count: uniform {1..max_k}, with k=1 override."""
    rng = np.random.default_rng(seed + step)
    if override_frac > 0 and rng.random() < override_frac:
        return 1
    return int(rng.integers(1, max_k + 1))


def load_packed(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data_dir = Path(data_dir)
    ids = np.load(data_dir / "input_ids.npy")
    labels = np.load(data_dir / "labels.npy")
    meta = json.loads((data_dir / "meta.json").read_text())
    return ids, labels, meta


def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: LoopedCausalLM, ids: np.ndarray, labels: np.ndarray, *, n_loops: int,
             batch_size: int, max_batches: int, device: torch.device,
             tbptt_depth: int = 0) -> float:
    model.eval()
    losses = []
    n = len(ids)
    for start in range(0, n, batch_size):
        if len(losses) >= max_batches:
            break
        idx = np.arange(start, min(start + batch_size, n))
        x = torch.as_tensor(ids[idx], device=device)
        y = torch.as_tensor(labels[idx], device=device)
        attn = torch.ones_like(x)
        out = model(x, attention_mask=attn, labels=y, n_loops=n_loops, tbptt_depth=tbptt_depth)
        losses.append(out["loss"].item())
    model.train()
    return float(np.mean(losses))


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, labels, meta = load_packed(cfg.data_dir)
    if cfg.seq_len != meta["seq_len"]:
        print(f"warning: cfg.seq_len={cfg.seq_len} != data seq_len={meta['seq_len']}")
    n = len(ids)
    n_val = min(max(20, int(0.05 * n)), 500)
    n_val = min(n_val, max(1, n // 5))
    perm = np.random.default_rng(cfg.seed).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ids, train_labels = ids[train_idx], labels[train_idx]
    val_ids, val_labels = ids[val_idx], labels[val_idx]
    print(f"data: {n} sequences ({n * meta['seq_len']} tokens); train={len(train_ids)} val={len(val_ids)}")

    model = LoopedCausalLM.from_base(
        cfg.base_model, cfg.n_input, cfg.n_recurrent, cfg.n_output,
        loop_norm=cfg.loop_norm, dtype=dtype,
        first_loop_passthrough=cfg.first_loop_passthrough,
        normalize_loop_output=cfg.normalize_loop_output,
        loop_residual=cfg.loop_residual,
        loop_gate_init=cfg.loop_gate_init,
        inject_input=cfg.inject_input,
    )
    apply_lora(model, rank=cfg.lora_rank, alpha=cfg.lora_alpha, blocks=cfg.lora_blocks)
    if cfg.grad_ckpt:
        model.enable_gradient_checkpointing()
    model.to(device).train()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    print(f"model: {cfg.base_model} partition={cfg.n_input}/{cfg.n_recurrent}/{cfg.n_output} "
          f"trainable={n_trainable / 1e6:.2f}M")

    params = [p for p in model.parameters() if p.requires_grad]
    gate_params = [model.loop_gate] if model.loop_gate is not None else []
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in params if id(p) not in gate_ids]
    optim = torch.optim.AdamW(
        [
            {"params": other_params, "lr": cfg.learning_rate},
            {"params": gate_params, "lr": cfg.learning_rate * cfg.gate_lr_mult},
        ],
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: lr_lambda(s, cfg.warmup_steps, cfg.max_steps)
    )

    use_wandb = cfg.wandb
    run = None
    if use_wandb:
        import wandb
        run = wandb.init(project="looped-upcycle", config=cfg.__dict__)

    train_log = (out_dir / "train.jsonl").open("a")
    eval_log = (out_dir / "eval.jsonl").open("a")

    order = np.random.permutation(len(train_ids))
    cursor = 0
    step = 0
    t0 = time.time()
    while step < cfg.max_steps:
        # gather a micro-batch
        if cursor + cfg.batch_size > len(order):
            order = np.random.permutation(len(train_ids))
            cursor = 0
        idx = order[cursor:cursor + cfg.batch_size]
        cursor += cfg.batch_size
        x = torch.as_tensor(train_ids[idx], device=device)
        y = torch.as_tensor(train_labels[idx], device=device)
        attn = torch.ones_like(x)

        k = sample_loop_count(step + cfg.seed_step_offset, cfg.max_k, cfg.override_frac, cfg.seed)
        out = model(x, attention_mask=attn, labels=y, n_loops=k, tbptt_depth=cfg.tbptt_depth)
        if not torch.isfinite(out["loss"]):
            print(f"warning: non-finite loss at step {step}; skipping")
            optim.zero_grad(set_to_none=True)
            step += 1
            continue
        loss = out["loss"] / cfg.grad_accum
        loss.backward()

        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

        if step % 10 == 0:
            gate = model.loop_gate.item() if model.loop_gate is not None else 0.0
            msg = (f"step {step} k={k} loss={out['loss'].item():.4f} gate={gate:.4f} "
                   f"lr={sched.get_last_lr()[0]:.2e} elapsed={time.time() - t0:.0f}s")
            print(msg)
            train_log.write(json.dumps({"step": step, "k": k, "loss": out["loss"].item(),
                                        "gate": gate, "lr": sched.get_last_lr()[0]}) + "\n")
            train_log.flush()
            if run:
                run.log({"train/loss": out["loss"].item(), "train/k": k,
                         "train/lr": sched.get_last_lr()[0], "step": step})

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            for kk in (1, 2, 3, 4):
                vl = evaluate(model, val_ids, val_labels, n_loops=kk, batch_size=cfg.batch_size,
                              max_batches=cfg.eval_batches, device=device, tbptt_depth=0)
                print(f"  eval k={kk} val_loss={vl:.4f}")
                eval_log.write(json.dumps({"step": step, "k": kk, "val_loss": vl}) + "\n")
                eval_log.flush()
                if run:
                    run.log({f"eval/val_loss_k{kk}": vl, "step": step})

        if cfg.save_every and step > 0 and step % cfg.save_every == 0:
            model.save_adapter(out_dir / f"adapter_step{step}")
            print(f"  saved adapter to {out_dir / f'adapter_step{step + 1}'}")

        step += 1

    model.save_adapter(out_dir / "adapter_final")
    (out_dir / "train_config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    print(f"done. final adapter -> {out_dir / 'adapter_final'}")
    train_log.close()
    eval_log.close()
    if run:
        run.finish()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B")
    p.add_argument("--data-dir", default="data/packed")
    p.add_argument("--out-dir", default="runs/looped")
    p.add_argument("--n-input", type=int, default=6)
    p.add_argument("--n-recurrent", type=int, default=16)
    p.add_argument("--n-output", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-k", type=int, default=4)
    p.add_argument("--override-frac", type=float, default=0.1)
    p.add_argument("--tbptt-depth", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=12181531)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--no-first-loop-passthrough", action="store_true")
    p.add_argument("--normalize-loop-output", action="store_true")
    p.add_argument("--no-loop-residual", action="store_true")
    p.add_argument("--loop-gate-init", type=float, default=0.0)
    p.add_argument("--no-inject-input", action="store_true")
    p.add_argument("--gate-lr-mult", type=float, default=10.0)
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    cfg = TrainConfig(
        base_model=args.base_model, data_dir=args.data_dir, out_dir=args.out_dir,
        n_input=args.n_input, n_recurrent=args.n_recurrent, n_output=args.n_output,
        seq_len=args.seq_len, batch_size=args.batch_size, grad_accum=args.grad_accum,
        max_k=args.max_k, override_frac=args.override_frac, tbptt_depth=args.tbptt_depth,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        max_steps=args.max_steps, eval_every=args.eval_every, eval_batches=args.eval_batches,
        save_every=args.save_every, seed=args.seed, device=args.device, dtype=args.dtype,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, grad_ckpt=not args.no_grad_ckpt,
        first_loop_passthrough=not args.no_first_loop_passthrough,
        normalize_loop_output=args.normalize_loop_output,
        loop_residual=not args.no_loop_residual,
        loop_gate_init=args.loop_gate_init,
        inject_input=not args.no_inject_input,
        gate_lr_mult=args.gate_lr_mult,
        wandb=args.wandb,
    )
    train(cfg)


if __name__ == "__main__":
    main()
