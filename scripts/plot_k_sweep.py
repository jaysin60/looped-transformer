"""Plot the k-sweep from a run's eval.jsonl."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="directory containing eval.jsonl")
    p.add_argument("--out", default=None, help="save PNG here instead of showing")
    args = p.parse_args()

    path = Path(args.run_dir) / "eval.jsonl"
    by_step: dict[int, dict[int, float]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        by_step[int(row["step"])][int(row["k"])] = float(row["val_loss"])

    if not by_step:
        print(f"no eval rows in {path}")
        return

    steps = sorted(by_step)
    ks = sorted({k for d in by_step.values() for k in d})
    print(f"run: {args.run_dir}")
    print(f"steps: {steps}")
    for step in steps:
        row = by_step[step]
        best = min(row, key=row.get)
        print(f"  step {step}: " + "  ".join(f"k={k} {row[k]:.4f}" for k in ks if k in row)
              + f"  (best k={best})")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for step in steps:
        xs = [k for k in ks if k in by_step[step]]
        ys = [by_step[step][k] for k in xs]
        ax.plot(xs, ys, marker="o", label=f"step {step}")
    ax.set_xlabel("loop count k")
    ax.set_ylabel("validation loss")
    ax.set_title("Looped upcycle: k-sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
