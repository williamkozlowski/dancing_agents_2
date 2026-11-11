"""show the latest and best evaluation results for each run."""
import glob
import os
import sys

import pandas as pd


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/sweep_next_interval"
    rows = []
    for csv in sorted(glob.glob(os.path.join(root, "*", "seed_*", "eval.csv"))):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        name, seed = csv.split(os.sep)[-3], csv.split(os.sep)[-2]
        last = df.iloc[-1]
        tapping = df[df.taps_per_episode > 0]
        best = tapping.loc[tapping.mean_reward.idxmax()] if len(tapping) else None
        log = os.path.join(os.path.dirname(csv), "train.log")
        crashed = os.path.exists(log) and "Traceback" in open(log).read()
        rows.append({
            "run": f"{name}/{seed}", "sim_h": round(last.sim_hours, 1),
            "reward": round(last.mean_reward, 3),
            "async_ms": round(last.mean_asynchrony_ms, 1),
            "taps/ep": round(last.taps_per_episode, 1),
            "sampled_rew": round(last.get("sampled_reward", float("nan")), 2),
            "ent": round(last.get("ent_coef", float("nan")), 4),
            "best_reward": None if best is None else round(best.mean_reward, 3),
            "best_at_h": None if best is None else round(best.sim_hours, 1),
            "status": "CRASHED" if crashed else "",
        })
    if not rows:
        sys.exit(f"no eval.csv files under {root}")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
