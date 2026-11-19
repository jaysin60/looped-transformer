"""Build the open-source agentic + coding SFT mix for the looped-model experiment.

Sources (all public, licenses recorded in data/manifest.json):
  - OpenCoder-LLM/opc-sft-stage2 (educational_instruct)   [MIT]          coding
  - open-thoughts/OpenThoughts-Agent-SFT-100K              [Apache-2.0]   agent trajectories
  - R2E-Gym/R2EGym-SFT-Trajectories                        [Apache-2.0]   SWE trajectories
  - nvidia/Nemotron-SFT-SWE-v3 (optional, one parquet shard) [CC BY 4.0] SWE trajectories

Nothing here is synthetic. The "unique" part is the mix, the difficulty metadata,
and the loop-curriculum routing applied downstream.

Usage:
    python data/build_mix.py inspect
    python data/build_mix.py build --out data/packed --tokenizer Qwen/Qwen2.5-Coder-1.5B \
        --seq-len 2048 --max-examples 2000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Source:
    name: str
    config: str | None
    split: str
    samples: int
    data_files: str | None = None
    license: str = ""
    adapter: str = "messages"
    note: str = ""


SOURCES: list[Source] = [
    Source("OpenCoder-LLM/opc-sft-stage2", "educational_instruct", "train", 20_000,
           license="MIT", adapter="opc_stage2",
           note="coding instruction -> solution (+ tests)"),
    Source("open-thoughts/OpenThoughts-Agent-SFT-100K", None, "train", 30_000,
           license="Apache-2.0", adapter="messages",
           note="agent trajectories (terminus-2)"),
    Source("R2E-Gym/R2EGym-SFT-Trajectories", None, "train", 3_231,
           license="Apache-2.0", adapter="messages",
           note="SWE agent trajectories"),
    # Uncomment to add a single 23GB-repo shard (keeps the download small):
    # Source("nvidia/Nemotron-SFT-SWE-v3", None, "train", 10_000,
    #        data_files="data/train-00000-of-00096.parquet",
    #        license="CC BY 4.0", adapter="messages", note="SWE trajectories"),
]


# ChatML template with the {% generation %} marker so that
# `return_assistant_tokens_mask=True` can identify assistant spans.
CHAT_TEMPLATE = """{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}{% endgeneration %}{% else %}{{ message['content'] }}{% endif %}<|im_end|>
{% endfor %}"""

# Long agentic system/tool messages can crowd out the assistant turns under
# right-truncation. Cap individual messages so a useful assistant span survives.
MAX_MESSAGE_CHARS = 1500


def load_source(src: Source):
    from datasets import load_dataset

    kwargs = {"split": src.split, "streaming": True}
    if src.data_files:
        kwargs["data_files"] = src.data_files
    if src.config:
        return load_dataset(src.name, src.config, **kwargs)
    return load_dataset(src.name, **kwargs)


def _messages(raw) -> list[dict]:
    out = []
    for m in raw or []:
        if isinstance(m, dict):
            role = m.get("role") or m.get("from")
            content = m.get("content") or m.get("value") or ""
        else:
            role, content = "user", str(m)
        if role in ("user", "assistant", "system", "tool") and content:
            content = str(content)
            if len(content) > MAX_MESSAGE_CHARS:
                content = content[:MAX_MESSAGE_CHARS] + " …"
            out.append({"role": role, "content": content})
    return out


def adapt(row: dict, adapter: str) -> list[dict] | None:
    if adapter == "messages":
        return _messages(row.get("conversations") or row.get("messages")) or None
    if adapter == "opc_stage2":
        instr = row.get("instruction") or row.get("problem")
        sol = row.get("output") or row.get("code") or row.get("solution")
        if not instr or not sol:
            return None
        return [{"role": "user", "content": str(instr)},
                {"role": "assistant", "content": str(sol)}]
    raise ValueError(f"unknown adapter {adapter!r}")


def difficulty(msgs: list[dict]) -> dict:
    return {
        "assistant_turns": sum(1 for m in msgs if m["role"] == "assistant"),
        "tool_turns": sum(1 for m in msgs if m["role"] == "tool"),
        "chars": sum(len(m["content"]) for m in msgs),
    }


def iter_examples(max_examples: int | None = None, seed: int = 12181531) -> Iterator[dict]:
    for src in SOURCES:
        n = src.samples
        if max_examples is not None:
            n = min(n, max_examples)
        ds = load_source(src).shuffle(seed=seed, buffer_size=10_000)
        taken = 0
        for row in ds:
            if taken >= n:
                break
            msgs = adapt(row, src.adapter)
            if not msgs or len(msgs) < 2 or msgs[-1]["role"] != "assistant":
                continue
            taken += 1
            yield {"messages": msgs, "source": src.name, "subset": src.config,
                   "difficulty": difficulty(msgs)}


def cmd_inspect(args) -> None:
    for src in SOURCES:
        print(f"\n=== {src.name} config={src.config} data_files={src.data_files} ===")
        try:
            ds = load_source(src)
            print("features:", ds.features)
            for i, row in enumerate(ds):
                if i >= 1:
                    break
                print("row:", json.dumps(
                    {k: (str(v)[:400] + "..." if len(str(v)) > 400 else str(v)) for k, v in row.items()},
                    ensure_ascii=False)[:1600])
        except Exception as e:
            print("ERROR:", type(e).__name__, str(e)[:500])


def cmd_build(args) -> None:
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eos = tok.eos_token_id
    seq_len = args.seq_len

    all_ids: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    buf_ids: list[int] = []
    buf_labels: list[int] = []
    n_examples = 0
    n_tokens = 0

    def flush() -> None:
        nonlocal buf_ids, buf_labels, n_tokens
        while len(buf_ids) >= seq_len:
            ids = np.asarray(buf_ids[:seq_len], dtype=np.int32)
            lab = np.asarray(buf_labels[:seq_len], dtype=np.int32)
            all_ids.append(ids)
            all_labels.append(lab)
            n_tokens += seq_len
            buf_ids = buf_ids[seq_len:]
            buf_labels = buf_labels[seq_len:]

    for ex in iter_examples(args.max_examples, seed=args.seed):
        n_examples += 1
        enc = tok.apply_chat_template(
            ex["messages"],
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_tensors="np",
            add_generation_prompt=False,
            truncation=True,
            max_length=seq_len,
            chat_template=CHAT_TEMPLATE,
        )
        ids = enc["input_ids"][0].astype(np.int64).tolist()
        mask = enc["assistant_masks"][0]
        labels = [int(t) if int(m) == 1 else -100 for t, m in zip(ids, mask)]
        if not any(l != -100 for l in labels):
            continue  # no assistant tokens -> no training signal
        buf_ids.extend(ids)
        buf_labels.extend(labels)
        if eos is not None:
            buf_ids.append(int(eos))
            buf_labels.append(-100)
        flush()
        if args.max_sequences and len(all_ids) >= args.max_sequences:
            break

    if buf_ids:
        # drop the short remainder; packing is more efficient without padding
        pass

    ids_arr = np.stack(all_ids) if all_ids else np.zeros((0, seq_len), dtype=np.int32)
    lab_arr = np.stack(all_labels) if all_labels else np.zeros((0, seq_len), dtype=np.int32)
    np.save(out / "input_ids.npy", ids_arr)
    np.save(out / "labels.npy", lab_arr)
    meta = {
        "tokenizer": args.tokenizer,
        "seq_len": seq_len,
        "num_sequences": int(ids_arr.shape[0]),
        "num_examples": n_examples,
        "num_tokens": int(ids_arr.shape[0] * seq_len),
        "seed": args.seed,
        "sources": [s.__dict__ for s in SOURCES],
        "chat_template": CHAT_TEMPLATE,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"examples={n_examples} sequences={ids_arr.shape[0]} tokens={ids_arr.shape[0] * seq_len}")
    print(f"wrote {out/'input_ids.npy'} and {out/'labels.npy'}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect").set_defaults(func=cmd_inspect)

    b = sub.add_parser("build")
    b.add_argument("--out", default="data/packed")
    b.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-1.5B")
    b.add_argument("--seq-len", type=int, default=2048)
    b.add_argument("--max-examples", type=int, default=None,
                   help="cap examples per source (for quick tests)")
    b.add_argument("--max-sequences", type=int, default=None)
    b.add_argument("--seed", type=int, default=12181531)
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
