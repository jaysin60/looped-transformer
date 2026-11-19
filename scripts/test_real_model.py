"""Load a real pretrained model, wrap it in LoopedCausalLM, and run on MPS."""

import argparse
import time

import torch

from looped.model import LoopedCausalLM, apply_lora


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B")
    p.add_argument("--n-input", type=int, default=5)
    p.add_argument("--n-recurrent", type=int, default=14)
    p.add_argument("--n-output", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    t0 = time.time()
    model = LoopedCausalLM.from_base(
        args.model,
        args.n_input,
        args.n_recurrent,
        args.n_output,
        loop_norm="rms",
        dtype=dtype,
    )
    apply_lora(model, rank=args.rank, alpha=float(args.rank), blocks=("recurrent",))
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()
    model.to(device).train()
    print(f"loaded {args.model} in {time.time() - t0:.1f}s")
    print("layers:", model.config.num_hidden_layers, "partition:",
          model.n_input, model.n_recurrent, model.n_output)
    print("trainable params:", sum(p.numel() for p in model.trainable_parameters()) / 1e6, "M")

    x = torch.randint(0, model.config.vocab_size, (args.batch, args.seq_len), device=device)
    labels = x.clone()

    for k in (1, 2, 4):
        model.zero_grad(set_to_none=True)
        t = time.time()
        out = model(x, labels=labels, n_loops=k, tbptt_depth=1)
        loss = out["loss"]
        loss.backward()
        dt = time.time() - t
        print(f"k={k} loss={loss.item():.4f} fwd+bwd={dt:.2f}s")
        model.zero_grad(set_to_none=True)

    model.eval()
    with torch.no_grad():
        for k in (1, 2, 4, 6):
            out = model(x, labels=labels, n_loops=k)
            print(f"eval k={k} loss={out['loss'].item():.4f}")


if __name__ == "__main__":
    main()
