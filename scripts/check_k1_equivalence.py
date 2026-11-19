"""Check that the looped wrapper at k=1 reproduces the base model exactly."""

import torch
from transformers import AutoModelForCausalLM

from looped.model import LoopedCausalLM


def main() -> None:
    torch.manual_seed(0)
    name = "Qwen/Qwen2.5-Coder-0.5B"
    base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    base.eval()

    x = torch.randint(0, base.config.vocab_size, (2, 128))
    y = x.clone()
    with torch.no_grad():
        base_loss = base(input_ids=x, labels=y).loss.item()

    looped = LoopedCausalLM(base, n_input=5, n_recurrent=14, n_output=5,
                            loop_norm="rms", first_loop_passthrough=True,
                            normalize_loop_output=False)
    looped.eval()
    with torch.no_grad():
        k1 = looped(x, labels=y, n_loops=1, tbptt_depth=0)["loss"].item()
        k2 = looped(x, labels=y, n_loops=2, tbptt_depth=0)["loss"].item()
        k4 = looped(x, labels=y, n_loops=4, tbptt_depth=0)["loss"].item()

    print(f"base loss     = {base_loss:.8f}")
    print(f"looped k=1    = {k1:.8f}  diff={abs(base_loss - k1):.2e}")
    print(f"looped k=2    = {k2:.8f}")
    print(f"looped k=4    = {k4:.8f}")
    assert abs(base_loss - k1) < 1e-3, "k=1 should match the base model"
    print("OK: k=1 is equivalent to the pretrained model")


if __name__ == "__main__":
    main()
