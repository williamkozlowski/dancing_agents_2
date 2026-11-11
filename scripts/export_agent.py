"""save an agent with its config, weights, and evaluation results."""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO  # noqa: E402

from rl_env.synchronization_env import SynchronizationEnv  # noqa: E402
from utils.evaluation import evaluate, make_env_config, run_episode, tap_for_beat  # noqa: E402


def phase_response(model, env_cfg, ioi=0.6, beat=6, shifts=(-150, -100, -50, 0, 50, 100, 150)):
    rows = []
    for s in shifts:
        cfg = make_env_config(env_cfg, perturb_beat=beat, perturb=s / 1000.0, perturb_mode="phase")
        r = run_episode(model, SynchronizationEnv(cfg), seed=1, options={"ioi": ioi})
        rows.append({"shift_ms": s, "induced_async_ms": tap_for_beat(r, beat - 1),
                     "next_async_ms": tap_for_beat(r, beat)})
    return rows


def continuation(model, env_cfg, episodes=100):
    env = SynchronizationEnv(make_env_config(env_cfg, sync_beats=10, continuation_beats=6))
    errs, n = [], []
    for i in range(episodes):
        r = run_episode(model, env, seed=500 + i)
        cont = [t for t in r["taps"] if t > r["cues"][-1] + r["ioi"] / 2]
        n.append(len(cont))
        if len(cont) > 1:
            errs.extend(np.abs(np.diff(cont) - r["ioi"]) * r["dt"] * 1000)
    return {"continuation_taps_per_episode": float(np.mean(n)),
            "mean_abs_iti_error_ms": float(np.mean(errs)) if errs else None}


def _clean(x):
    """JSON has no NaN: write null for undefined values (e.g. no taps)."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--note", default="", help="free-text description stored in metrics.json")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["environment"]
    model = RecurrentPPO.load(args.model, device="cpu")

    metrics = {
        "note": args.note,
        "reward_type": env_cfg["reward_type"],
        "steady": evaluate(model, env_cfg, episodes=100, seed=777),
        "jitter_10ms": evaluate(model, make_env_config(env_cfg, jitter=0.01), episodes=100, seed=777),
        "beats_100": evaluate(model, make_env_config(env_cfg, sync_beats=100), episodes=10, seed=777),
        "phase_response_600ms": phase_response(model, env_cfg),
        "continuation": continuation(model, env_cfg),
    }

    os.makedirs(args.out, exist_ok=True)
    shutil.copyfile(args.model if args.model.endswith(".zip") else args.model + ".zip",
                    os.path.join(args.out, "model.zip"))
    torch.save(model.policy.state_dict(), os.path.join(args.out, "policy_weights.pt"))
    with open(os.path.join(args.out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(_clean(metrics), f, indent=2)

    s = metrics["steady"]
    print(f"Exported to {args.out}")
    print(f"  steady: reward {s['mean_reward']:+.3f}, mean async {s['mean_asynchrony_ms']:+.1f} ms, "
          f"{s['taps_per_episode']:.1f} evaluated taps/episode")
    print(f"  jitter 10 ms: reward {metrics['jitter_10ms']['mean_reward']:+.3f}, "
          f"mean async {metrics['jitter_10ms']['mean_asynchrony_ms']:+.1f} ms")
    print("  phase response (shift: induced -> next async, ms):",
          ", ".join(f"{r['shift_ms']:+d}: {r['induced_async_ms']} -> {r['next_async_ms']}"
                    for r in metrics["phase_response_600ms"]))
    print("  continuation:", metrics["continuation"])


if __name__ == "__main__":
    main()
