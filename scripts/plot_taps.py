# made by claude: 2026-11-19 - Will

"""plot sampled taps per episode over training from an eval.csv."""
import argparse
import os
import sys

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="path to a run's eval.csv")
    p.add_argument("--out", default="sampled_taps.png", help="output image path")
    p.add_argument("--hours", action="store_true",
                   help="use simulated hours on the x-axis instead of timesteps")
    p.add_argument("--deterministic", action="store_true",
                   help="also plot deterministic taps_per_episode")
    p.add_argument("--beats", type=float, default=7.0,
                   help="evaluated beats per episode, drawn as a reference line "
                        "(default 7 = 10 cues minus 3 unscored); 0 to hide")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "sampled_taps_per_episode" not in df.columns:
        sys.exit(f"{args.csv} has no sampled_taps_per_episode column")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_col, x_label = ("sim_hours", "Simulated hours") if args.hours else ("timesteps", "Timesteps")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df[x_col], df["sampled_taps_per_episode"], marker="o", ms=3, label="sampled policy")
    if args.deterministic and "taps_per_episode" in df.columns:
        ax.plot(df[x_col], df["taps_per_episode"], marker="s", ms=3, ls="--",
                label="deterministic (P_tap > 0.5)")
    if args.beats > 0:
        ax.axhline(args.beats, color="k", lw=0.8, ls=":", label=f"{args.beats:g} beats/episode")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Taps per episode")
    ax.set_title(os.path.dirname(os.path.abspath(args.csv)).split(os.sep)[-2:].__str__())
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()