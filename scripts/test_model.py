"""test an agent with steady, jittered, shifted, or silent beats."""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO  # noqa: E402

from rl_env.synchronization_env import SynchronizationEnv  # noqa: E402
from utils.evaluation import make_env_config, run_episode, tap_for_beat  # noqa: E402


def print_single_trial(result):
    dt_ms = result["dt"] * 1000
    print(f"IOI: {result['ioi'] * dt_ms:.0f} ms   episode reward: {result['reward']:+.3f}")
    print("Cue times (ms):", (np.array(result["cues"]) * dt_ms).astype(int).tolist())
    print("Tap times (ms):", (np.array(result["taps"]) * dt_ms).astype(int).tolist())
    print("Evaluated tap-cue asynchronies (ms):", [round(a) for a in result["eval_async"]])
    if result["eval_async"]:
        print(f"Mean asynchrony (evaluated taps): {np.mean(result['eval_async']):+.1f} ms")


def steady(model, env_cfg, episodes, seed):
    env = SynchronizationEnv(env_cfg)
    rewards, ep_means, pooled = [], [], []
    for i in range(episodes):
        r = run_episode(model, env, seed=seed + i)
        rewards.append(r["reward"])
        if r["eval_async"]:
            ep_means.append(np.mean(r["eval_async"]))
            pooled.extend(r["eval_async"])
    pooled = np.asarray(pooled)
    print(f"Evaluation over {episodes} episodes, IOI {env.min_ioi*1000:.0f}-{env.max_ioi*1000:.0f} ms, "
          f"jitter {env.jitter*1000:.0f} ms, {env.sync_beats} cues:")
    print(f"  Mean episode reward:  {np.mean(rewards):+.3f} (SD {np.std(rewards):.3f})")
    if pooled.size:
        print(f"  Mean asynchrony:      {np.mean(ep_means):+.1f} ms (SD across episodes {np.std(ep_means):.1f})")
        print(f"  Mean |asynchrony|:    {np.mean(np.abs(pooled)):.1f} ms")
        print(f"  Pooled tap SD:        {np.std(pooled):.1f} ms   ({pooled.size / episodes:.1f} evaluated taps/episode)")
    else:
        print("  No evaluated taps.")


def phase_response(model, env_cfg, ioi, beat, shifts_ms, seed):
    """shift the metronome and measure tap timing on this beat and the next."""
    print(f"Phase-shift test: IOI {ioi*1000:.0f} ms, cue {beat} shifted")
    print(f"{'shift (ms)':>10} {'induced async':>14} {'next async':>11} {'correction':>11}")
    rows = []
    for s in shifts_ms:
        cfg = make_env_config(env_cfg, perturb_beat=beat, perturb=s / 1000.0, perturb_mode="phase")
        r = run_episode(model, SynchronizationEnv(cfg), seed=seed, options={"ioi": ioi})
        a0, a1 = tap_for_beat(r, beat - 1), tap_for_beat(r, beat)
        corr = (1 - a1 / a0) if (a0 not in (None, 0) and a1 is not None) else float("nan")
        rows.append((s, a0, a1, corr))
        fmt = lambda v: f"{v:+.0f}" if v is not None else "  -"
        print(f"{s:>+10.0f} {fmt(a0):>14} {fmt(a1):>11} {corr:>11.2f}")
    return rows


def continuation(model, env_cfg, episodes, seed, sync_beats=10, cont_beats=6):
    cfg = make_env_config(env_cfg, sync_beats=sync_beats, continuation_beats=cont_beats)
    env = SynchronizationEnv(cfg)
    errs, n_cont = [], []
    for i in range(episodes):
        r = run_episode(model, env, seed=seed + i)
        last_cue = r["cues"][-1]
        cont = [t for t in r["taps"] if t > last_cue + r["ioi"] / 2]
        n_cont.append(len(cont))
        if len(cont) > 1:
            errs.extend(np.abs(np.diff(cont) - r["ioi"]) * r["dt"] * 1000)
        if i == 0:
            print_single_trial(r)
            print("Continuation ITIs (ms):", (np.diff(cont) * r["dt"] * 1000).astype(int).tolist())
    print(f"\nSynchronization-continuation over {episodes} episodes "
          f"({sync_beats} cues + {cont_beats} silent beats):")
    print(f"  Continuation taps per episode: {np.mean(n_cont):.1f} (expected ~{cont_beats})")
    if errs:
        print(f"  Mean |ITI - IOI|:  {np.mean(errs):.1f} +/- {np.std(errs):.1f} ms")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--ioi", type=float, help="fixed IOI in seconds")
    p.add_argument("--jitter", type=float, help="cue jitter SD in seconds (e.g. 0.03)")
    p.add_argument("--beats", type=int, help="number of audible cues (e.g. 100)")
    p.add_argument("--perturb-ms", type=float, dest="perturb_ms")
    p.add_argument("--perturb-beat", type=int, default=6, dest="perturb_beat")
    p.add_argument("--perturb-mode", default="phase", choices=["phase", "event"], dest="perturb_mode")
    p.add_argument("--prc", action="store_true", help="phase-response curve at 600 ms IOI")
    p.add_argument("--prc-ioi", type=float, default=0.6)
    p.add_argument("--continuation", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = make_env_config(cfg["environment"], jitter=args.jitter, sync_beats=args.beats)
    if args.ioi:
        env_cfg["min_ioi"] = env_cfg["max_ioi"] = args.ioi
    if args.perturb_ms is not None:
        env_cfg.update(perturb_beat=args.perturb_beat, perturb=args.perturb_ms / 1000.0,
                       perturb_mode=args.perturb_mode)
    model = RecurrentPPO.load(args.model, device="cpu")

    if args.prc:
        phase_response(model, env_cfg, args.prc_ioi, args.perturb_beat,
                       [-150, -100, -50, -20, 0, 20, 50, 100, 150], args.seed)
    elif args.continuation:
        continuation(model, env_cfg, args.episodes, args.seed)
    elif args.episodes <= 1:
        print_single_trial(run_episode(model, SynchronizationEnv(env_cfg), seed=args.seed))
    else:
        steady(model, env_cfg, args.episodes, args.seed)


if __name__ == "__main__":
    main()
