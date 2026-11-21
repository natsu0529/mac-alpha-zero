"""Plot search scores and optional Elo against single-Mac training time."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(log_path, output, ratings_path=None):
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    levels = sorted(
        {
            int(key.split()[-1])
            for row in rows
            for key in row["scores"]
            if key.startswith("search vs Edax level ")
        }
    )
    if not levels:
        raise ValueError("the log has no search scores against Edax")
    fig, axes = plt.subplots(
        2 if ratings_path else 1,
        figsize=(7.5, 4.8 if ratings_path else 3.5),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    ax = axes[0, 0]
    for level in levels:
        key = f"search vs Edax level {level}"
        points = [(row["minutes"] / 60, row["scores"][key]) for row in rows if key in row["scores"]]
        ax.plot(*zip(*points, strict=True), marker=".", linewidth=1.5, label=f"Edax level {level}")
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1)
    ax.set(
        ylabel="Match score", ylim=(-0.03, 1.03), title="Othello search strength during training"
    )
    ax.legend(frameon=False, ncols=min(len(levels), 3))
    if ratings_path:
        ratings = [json.loads(line) for line in ratings_path.read_text().splitlines()]
        if [(r["iteration"], r["minutes"]) for r in ratings] != [
            (r["iteration"], r["minutes"]) for r in rows
        ]:
            raise ValueError(
                "ratings must cover the same iterations and minutes as the training log"
            )
        axes[1, 0].plot(
            [r["minutes"] / 60 for r in ratings],
            [r["elo"] for r in ratings],
            color="#65379b",
            marker=".",
        )
        axes[1, 0].set_ylabel("Relative Elo")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Training hours on one GPU (evaluation excluded)")
    fig.savefig(output, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ratings", type=Path, help="JSONL output from checkpoint_elo.py")
    args = parser.parse_args()
    plot(args.log, args.output, args.ratings)


if __name__ == "__main__":
    main()
