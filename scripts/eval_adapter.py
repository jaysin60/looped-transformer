"""Evaluate a saved looped adapter at several loop counts."""

import argparse

import numpy as np
import torch

from looped.model import LoopedCausalLM
from looped.trainer import evaluate, load_packed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--data-dir", default="data/packed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--ks", default="1,2,3,4")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.05)
    args = p.parse_args()

    device = torch.device(args.device)
    ids, labels, meta = load_packed(args.data_dir)
    n_val = min(max(50, int(args.val_frac * len(ids))), 1000)
    perm = np.random.default_rng(12181531).permutation(len(ids))
    val_ids, val_labels = ids[perm[:n_val]], labels[perm[:n_val]]

    model = LoopedCausalLM.load_adapter(args.adapter, dtype=args.dtype).to(device)
    model.eval()
    print(f"adapter={args.adapter}")
    print(f"config={model.loop_config()}")
    print(f"val_sequences={len(val_ids)}")
    if model.loop_gate is not None:
        print(f"gate={model.loop_gate.item():.5f}")

    for k in [int(x) for x in args.ks.split(",")]:
        loss = evaluate(model, val_ids, val_labels, n_loops=k, batch_size=args.batch_size,
                        max_batches=args.max_batches, device=device, tbptt_depth=0)
        print(f"k={k} val_loss={loss:.4f}")


if __name__ == "__main__":
    main()
