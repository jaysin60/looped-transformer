"""Looped (latent-recurrent) causal LM by upcycling a pretrained HF model.

The architecture follows Dory1.0-4B (arXiv:2611.03998):

    x = input_block(embed(tokens))          # fixed representation
    s = x
    for t in 1..k:
        u = norm(s + x)                      # re-inject the input, normalize
        s = norm(recurrent_block(u))         # shared weights, normalize
    h = output_block(s)
    logits = lm_head(norm(h))

The recurrent block is the *same* module applied k times. Training uses
truncated backpropagation through loops (TBPTL): only the last `tbptt_depth`
loops are differentiated.

This module is a plain `nn.Module` (not a PreTrainedModel) so it can wrap any
Llama/Qwen2-style model. LoRA is implemented directly (no peft dependency).
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

class L2Norm(nn.Module):
    """Dory `justnorm`: unit L2 norm over the last dim, accumulated in fp32."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), p=2, dim=-1).to(x.dtype)


def make_loop_norm(kind: str, hidden_size: int) -> nn.Module:
    if kind == "rms":
        return nn.RMSNorm(hidden_size, eps=1e-6)
    if kind == "l2":
        return L2Norm()
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown loop_norm {kind!r}")


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Frozen base linear plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(base.in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, base.out_features))
        nn.init.normal_(self.A, std=0.02)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.A.to(x.dtype)
        B = self.B.to(x.dtype)
        return self.base(x) + (x @ A @ B) * self.scaling


DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    targets: Iterable[str] = DEFAULT_LORA_TARGETS,
    blocks: Iterable[str] = ("recurrent",),
) -> None:
    """Replace targeted Linears in the chosen blocks with LoRALinear."""
    targets = set(targets)
    blocks = list(blocks)
    replaced = 0
    for block_name in blocks:
        block = getattr(model, f"{block_name}_layers", None)
        if block is None:
            raise ValueError(f"model has no {block_name}_layers")
        for layer in block:
            for submodule in list(layer.modules()):
                for name, child in list(submodule.named_children()):
                    if name in targets and isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
                        setattr(submodule, name, LoRALinear(child, rank, alpha))
                        replaced += 1
    # freeze everything, then unfreeze LoRA parameters
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.A.requires_grad = True
            module.B.requires_grad = True
    if hasattr(model, "unfreeze_loop_norms"):
        model.unfreeze_loop_norms()
    model._lora_config = {
        "rank": rank,
        "alpha": alpha,
        "targets": sorted(targets),
        "blocks": blocks,
    }
    print(f"apply_lora: replaced {replaced} linears (rank={rank}, alpha={alpha})")


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]


# ---------------------------------------------------------------------------
# looped model
# ---------------------------------------------------------------------------

class LoopedCausalLM(nn.Module):
    """Wrap a pretrained Llama/Qwen2 causal LM with a shared recurrent block."""

    def __init__(
        self,
        base_model: nn.Module,
        n_input: int,
        n_recurrent: int,
        n_output: int | None = None,
        loop_norm: str = "rms",
        first_loop_passthrough: bool = True,
        normalize_loop_output: bool = False,
        loop_residual: bool = True,
        loop_gate_init: float = 0.0,
        inject_input: bool = True,
    ):
        super().__init__()
        self.config = base_model.config
        self.base_model_name = getattr(base_model, "name_or_path", None) or self.config._name_or_path

        # take ownership of the pieces we need
        self.embed_tokens = base_model.model.embed_tokens
        self.rotary_emb = base_model.model.rotary_emb
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head

        layers = list(base_model.model.layers)
        n_total = len(layers)
        if n_output is None:
            n_output = n_total - n_input - n_recurrent
        if n_input + n_recurrent + n_output != n_total:
            raise ValueError(
                f"partition {n_input}+{n_recurrent}+{n_output} != {n_total} layers"
            )
        self.n_input = n_input
        self.n_recurrent = n_recurrent
        self.n_output = n_output
        self.input_layers = nn.ModuleList(layers[:n_input])
        self.recurrent_layers = nn.ModuleList(layers[n_input:n_input + n_recurrent])
        self.output_layers = nn.ModuleList(layers[n_input + n_recurrent:])

        self.loop_norm_type = loop_norm
        self.first_loop_passthrough = first_loop_passthrough
        self.normalize_loop_output = normalize_loop_output
        self.loop_residual = loop_residual
        self.loop_gate_init = loop_gate_init
        self.inject_input = inject_input
        base_dtype = self.embed_tokens.weight.dtype
        self.loop_norm_in = make_loop_norm(loop_norm, self.config.hidden_size).to(base_dtype)
        self.loop_norm_out = make_loop_norm(loop_norm, self.config.hidden_size).to(base_dtype)
        self.loop_gate = (
            nn.Parameter(torch.full((1,), float(loop_gate_init), dtype=torch.float32))
            if loop_residual else None
        )

        self.gradient_checkpointing = False

        for p in self.parameters():
            p.requires_grad = False
        self.unfreeze_loop_norms()

    def unfreeze_loop_norms(self) -> None:
        """Loop-boundary norms and the loop gate are small but important."""
        for m in (self.loop_norm_in, self.loop_norm_out):
            for p in m.parameters():
                p.requires_grad = True
        if self.loop_gate is not None:
            self.loop_gate.requires_grad = True

    # -- construction -------------------------------------------------------

    @classmethod
    def from_base(
        cls,
        model_name: str,
        n_input: int,
        n_recurrent: int,
        n_output: int | None = None,
        loop_norm: str = "rms",
        dtype: torch.dtype | str = "bfloat16",
        device_map: str | dict | None = None,
        trust_remote_code: bool = False,
        first_loop_passthrough: bool = True,
        normalize_loop_output: bool = False,
        loop_residual: bool = True,
        loop_gate_init: float = 0.0,
        inject_input: bool = True,
    ) -> "LoopedCausalLM":
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        return cls(
            base, n_input, n_recurrent, n_output, loop_norm,
            first_loop_passthrough=first_loop_passthrough,
            normalize_loop_output=normalize_loop_output,
            loop_residual=loop_residual,
            loop_gate_init=loop_gate_init,
            inject_input=inject_input,
        )

    # -- helpers ------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def loop_config(self) -> dict:
        return {
            "base_model": self.base_model_name,
            "n_input": self.n_input,
            "n_recurrent": self.n_recurrent,
            "n_output": self.n_output,
            "loop_norm": self.loop_norm_type,
            "first_loop_passthrough": self.first_loop_passthrough,
            "normalize_loop_output": self.normalize_loop_output,
            "loop_residual": self.loop_residual,
            "loop_gate_init": self.loop_gate_init,
            "inject_input": self.inject_input,
        }

    def _run_block(self, layers, hidden_states, attention_mask, position_embeddings, position_ids):
        for layer in layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
            )
        return hidden_states

    def _run_recurrent(self, hidden_states, attention_mask, position_embeddings, position_ids):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._run_block,
                self.recurrent_layers,
                hidden_states,
                attention_mask,
                position_embeddings,
                position_ids,
                use_reentrant=False,
            )
        return self._run_block(
            self.recurrent_layers, hidden_states, attention_mask, position_embeddings, position_ids
        )

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        n_loops: int = 2,
        tbptt_depth: int = 1,
        logits_to_keep: int = 0,
    ) -> dict:
        inputs_embeds = self.embed_tokens(input_ids.long())
        batch, seq_len = inputs_embeds.shape[:2]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        h = self._run_block(
            self.input_layers, inputs_embeds, causal_mask, position_embeddings, position_ids
        )
        x = h
        s = x

        # TBPTL: loops before the last `tbptt_depth` run without grad.
        n_early = 0 if tbptt_depth <= 0 else max(0, n_loops - tbptt_depth)
        for t in range(n_loops):
            with (torch.no_grad() if t < n_early else nullcontext()):
                # Upcycling: the first pass is the untouched pretrained stack,
                # so k=1 reproduces the base model exactly. Later passes use the
                # Dory recurrence: u = norm(s + x), s = F(u).
                if t == 0 and self.first_loop_passthrough:
                    u = x
                else:
                    u = self.loop_norm_in(s + x if self.inject_input else s)
                delta = self._run_recurrent(u, causal_mask, position_embeddings, position_ids)
                if t > 0 and self.normalize_loop_output:
                    delta = self.loop_norm_out(delta)
                if t == 0 or not self.loop_residual:
                    s = delta
                else:
                    s = s + self.loop_gate.to(delta.dtype) * delta

        h = self._run_block(
            self.output_layers, s, causal_mask, position_embeddings, position_ids
        )
        h = self.norm(h)
        logits = self.lm_head(h)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1)).float()
            shift_labels = labels[:, 1:].reshape(-1).long()
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        return {"loss": loss, "logits": logits}

    # -- generation (naive, no KV cache; fine for short eval) --------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        n_loops: int = 2,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.LongTensor:
        for _ in range(max_new_tokens):
            out = self(input_ids, n_loops=n_loops)["logits"][:, -1]
            if temperature <= 0:
                next_id = out.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(out / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = sorted_probs.cumsum(-1)
                    remove = cum - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(-1, keepdim=True)
                next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).all()):
                break
        return input_ids

    # -- persistence --------------------------------------------------------

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.named_parameters() if v.requires_grad}

    def save_adapter(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "loop_config": self.loop_config(),
                "lora_config": getattr(self, "_lora_config", None),
                "trainable_state_dict": self.trainable_state_dict(),
            },
            path / "adapter.pt",
        )
        (path / "loop_config.json").write_text(json.dumps(self.loop_config(), indent=2))

    @classmethod
    def load_adapter(cls, path: str | Path, dtype: torch.dtype | str = "bfloat16") -> "LoopedCausalLM":
        path = Path(path)
        payload = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
        loop = payload["loop_config"]
        model = cls.from_base(
            loop["base_model"],
            loop["n_input"],
            loop["n_recurrent"],
            loop["n_output"],
            loop["loop_norm"],
            dtype=dtype,
            first_loop_passthrough=loop.get("first_loop_passthrough", True),
            normalize_loop_output=loop.get("normalize_loop_output", False),
            loop_residual=loop.get("loop_residual", True),
            loop_gate_init=loop.get("loop_gate_init", 0.0),
            inject_input=loop.get("inject_input", True),
        )
        lora = payload.get("lora_config")
        if lora:
            apply_lora(
                model,
                rank=lora["rank"],
                alpha=lora["alpha"],
                targets=lora.get("targets", DEFAULT_LORA_TARGETS),
                blocks=lora.get("blocks", ("input", "recurrent", "output")),
            )
        missing, unexpected = model.load_state_dict(payload["trainable_state_dict"], strict=False)
        if unexpected:
            print(f"warning: unexpected adapter keys: {unexpected[:5]}")
        return model
