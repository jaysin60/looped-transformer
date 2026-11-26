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
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#98a5b8",
        "axes.labelcolor": "#344054",
        "text.color": "#24324a",
        "xtick.color": "#667085",
        "ytick.color": "#667085",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="#f5f7fb")
    colors = ["#6d4bd0", "#087f8c"]

    for color, (label, stem) in zip(colors, RUNS.items()):
        evals = read_jsonl(LOGS / f"{stem}_eval.jsonl")
        by_k = {row["k"]: row["val_loss"] for row in evals}
        ks = sorted(by_k)
        base = by_k[ks[0]]
        deltas = [(by_k[k] - base) * 1e3 for k in ks]
        ax1.plot(ks, deltas, marker="o", linewidth=2, markersize=5, color=color, label=label)

        train = read_jsonl(LOGS / f"{stem}_train.jsonl")
        steps = [r["step"] for r in train]
        gates = [r["gate"] for r in train]
        ax2.plot(steps, gates, linewidth=1.8, color=color, label=label)

    for ax in (ax1, ax2):
        ax.set_facecolor("#f5f7fb")
        ax.axhline(0, color="#8996aa", linewidth=0.8)
        ax.grid(True, color="#cbd5e1", linewidth=0.7, linestyle="-", alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
        ax.title.set_color("#3f4d9b")
        ax.title.set_fontfamily("DejaVu Sans")
        ax.title.set_fontstyle("normal")
        ax.title.set_fontweight(600)

    ax1.set_xlabel("loop count k · same checkpoint")
    ax1.set_ylabel("Δ validation loss vs k=1  (×10⁻³)")
    ax1.set_title("Does looping help?  negative = lower loss")
    ax1.set_xticks([1, 2, 3, 4])

    ax2.set_xlabel("training micro-batch")
    ax2.set_ylabel("scalar loop gate")
    ax2.set_title("The refinement stays near zero")

    fig.suptitle("Looped upcycle of Qwen2.5-Coder-0.5B on agentic + coding data",
                 fontsize=13, fontfamily="DejaVu Sans", color="#3f4d9b")
    fig.text(0.5, 0.005,
             "k=1 uses one pass and matched the pretrained base at initialization. "
             "200/120 micro-batches = 0.410M/0.246M token presentations; in-training eval uses 8 held-out sequences. "
             "Differences ~1e-3 are diagnostics, not robust gains.",
             ha="center", fontsize=7.3, color="#667085")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    out = ROOT / "results" / "k_sweep.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
