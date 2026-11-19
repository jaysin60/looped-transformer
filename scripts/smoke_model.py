"""Fast structural smoke test for LoopedCausalLM (no downloads required)."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from looped.model import LoopedCausalLM, apply_lora, lora_parameters


def build_tiny() -> LoopedCausalLM:
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    base = Qwen2ForCausalLM(config)
    model = LoopedCausalLM(base, n_input=2, n_recurrent=4, n_output=2, loop_norm="rms")
    apply_lora(model, rank=4, alpha=4.0, blocks=("recurrent",))
    return model


def main() -> None:
    torch.manual_seed(0)
    model = build_tiny().train()
    model.enable_gradient_checkpointing()
    print("loop config:", model.loop_config())
    print("trainable params:", sum(p.numel() for p in model.trainable_parameters()))
    print("lora params:", len(lora_parameters(model)))

    x = torch.randint(0, 256, (2, 32))
    labels = x.clone()

    for n_loops in (1, 2, 4):
        for tbptt_depth in (0, 1):
            model.zero_grad(set_to_none=True)
            out = model(x, labels=labels, n_loops=n_loops, tbptt_depth=tbptt_depth)
            loss = out["loss"]
            loss.backward()
            grads = [p.grad.norm().item() for p in model.trainable_parameters() if p.grad is not None]
            print(
                f"k={n_loops} tbptt={tbptt_depth} loss={loss.item():.4f} "
                f"logits={tuple(out['logits'].shape)} n_grads={len(grads)} "
                f"grad_norm_sum={sum(grads):.4f}"
            )

    # k-sweep eval (no grad)
    model.eval()
    with torch.no_grad():
        for n_loops in (1, 2, 3, 4, 6):
            out = model(x, labels=labels, n_loops=n_loops)
            print(f"eval k={n_loops} loss={out['loss'].item():.4f}")

    print("OK")


if __name__ == "__main__":
    main()
