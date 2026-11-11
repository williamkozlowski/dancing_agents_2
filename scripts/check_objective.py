"""compare simple tapping strategies under each reward mode."""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_env.synchronization_env import REWARD_TYPES, SynchronizationEnv  # noqa: E402


def _land_at(env, offset=0):
    """True if a tap initiated now lands ``offset`` steps from a scored beat."""
    return (env.current_step + env.tap_delay - offset) in env.virtual_cue_times[env.entrainment_beats:]


def reference_policies(rng):
    return {
        "never tap": lambda e: False,
        "tap once in entrainment": lambda e: e.current_step == e.cue_times[1],
        "random taps (p=0.02/step)": lambda e: rng.random() < 0.02,
        "reactive (tap on hearing a cue)": lambda e: e.current_step in e._cue_set,
        "predictive, on time": lambda e: _land_at(e, 0),
        "predictive, 20 ms early": lambda e: _land_at(e, -2),
        "predictive, 20 ms late": lambda e: _land_at(e, +2),
        "on time + extra tap 100 ms late": lambda e: _land_at(e, 0) or _land_at(e, 10),
    }


def rollout(env, policy, gamma, seed):
    env.reset(seed=seed)
    rewards, done = [], False
    while not done:
        _, r, done, _, _ = env.step(int(policy(env)))
        rewards.append(r)
    rewards = np.asarray(rewards)
    g, togo = 0.0, np.empty_like(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        togo[t] = g
    return rewards.sum(), togo.mean(), rewards.max()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--gamma", type=float, default=None, help="default: rl_algorithm.gamma")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--reward-type", default=None, help="default: all four schemes")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    gamma = args.gamma if args.gamma is not None else float(cfg["rl_algorithm"].get("gamma", 0.99))
    schemes = [args.reward_type] if args.reward_type else list(REWARD_TYPES)
    ok = True
    for rt in schemes:
        env = SynchronizationEnv({**cfg["environment"], "reward_type": rt})
        print(f"\n== {rt}   (gamma = {gamma}, {args.episodes} episodes at random tempi)")
        print(f"   {'policy':34s} {'episode reward':>15s} {'discounted value':>17s}")
        scores, max_r = {}, -np.inf
        for name, pol in reference_policies(np.random.default_rng(0)).items():
            res = np.array([rollout(env, pol, gamma, s) for s in range(args.episodes)])
            scores[name] = res[:, 1].mean()
            max_r = max(max_r, res[:, 2].max())
            print(f"   {name:34s} {res[:, 0].mean():>+15.3f} {res[:, 1].mean():>+17.4f}")
        best = max(scores, key=scores.get)
        print(f"   -> best under the discounted objective: {best}")
        if max_r > 0:
            print(f"   !! positive reward found ({max_r:+.3f}); rewards must be <= 0")
            ok = False
        if best != "predictive, on time":
            print("   !! on-time predictive tapping is not the best reference policy under this objective")
            ok = False
        if scores["never tap"] >= scores["random taps (p=0.02/step)"]:
            print("   !  never tapping beats random tapping: exploration will be "
                  "pushed toward silence (raise gamma)")
    print("\nOK" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
