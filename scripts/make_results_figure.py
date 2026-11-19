"""Build the results figure from the committed evaluation logs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"

RUNS = {
    "5/14/5 (0.5B, step 100)": "medium_200",
    "6/10/8 (0.5B, step 60)": "medium_smallrec",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for label, stem in RUNS.items():
        evals = read_jsonl(LOGS / f"{stem}_eval.jsonl")
        by_k = {row["k"]: row["val_loss"] for row in evals}
        ks = sorted(by_k)
        base = by_k[ks[0]]
        deltas = [(by_k[k] - base) * 1e3 for k in ks]
        ax1.plot(ks, deltas, marker="o", linewidth=2, label=label)

        train = read_jsonl(LOGS / f"{stem}_train.jsonl")
        steps = [r["step"] for r in train]
        gates = [r["gate"] for r in train]
        ax2.plot(steps, gates, linewidth=1.8, label=label)

    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("loop count k")
    ax1.set_ylabel("Δ validation loss vs k=1  (×10⁻³)")
    ax1.set_title("Does looping help? (negative = loop helps)")
    ax1.set_xticks([1, 2, 3, 4])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("loop gate value")
    ax2.set_title("Learned loop gate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("Looped upcycle of Qwen2.5-Coder-0.5B on agentic+coding data", fontsize=11)
    fig.text(0.5, 0.005,
             "k=1 is exactly the pretrained model. 0.5B LoRA, ~1.6M tokens, M6 MPS. "
             "Gains are within ~1e-3; the loop is not yet earning its keep at this budget.",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = ROOT / "results" / "k_sweep.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
