"""clone an on-time tapper and check how further learning changes it."""
import argparse
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_env.synchronization_env import SynchronizationEnv  # noqa: E402
from scripts.train import build_model, load_config, make_vec_env  # noqa: E402
from utils import fast_lstm  # noqa: E402
from utils.evaluation import evaluate  # noqa: E402

fast_lstm.enable()


def teacher_episodes(env_cfg, n, gamma, seed):
    """Episodes of a scripted tapper whose taps land exactly on every scored beat."""
    env = SynchronizationEnv(env_cfg)
    if env.observation_space.shape != (1,):
        raise SystemExit("stability_check.py supports only a single metronome input "
                         "(efference_input must be false)")
    X, A, G = [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        targets = set(env.virtual_cue_times[env.entrainment_beats:])
        xs, acts, rews, done = [], [], [], False
        while not done:
            a = int(env.current_step + env.tap_delay in targets)
            xs.append(obs[0]); acts.append(a)
            obs, r, done, _, _ = env.step(a)
            rews.append(r)
        g, ret = 0.0, np.empty(len(rews))
        for t in range(len(rews) - 1, -1, -1):
            g = rews[t] + gamma * g
            ret[t] = g
        X.append(np.asarray(xs, np.float32)); A.append(np.asarray(acts)); G.append(ret)
    return X, A, G


def pad(seqs, dtype):
    T = max(len(s) for s in seqs)
    out = np.zeros((T, len(seqs)), dtype=dtype)
    mask = np.zeros((T, len(seqs)), dtype=np.float32)
    for j, s in enumerate(seqs):
        out[: len(s), j] = s
        mask[: len(s), j] = 1.0
    return th.as_tensor(out), th.as_tensor(mask)


def clone(model, X, A, G, iters, batch, lr, seed, tap_weight=10.0):
    pol = model.policy
    params = (list(pol.lstm_actor.parameters()) + list(pol.action_net.parameters())
              + list(pol.lstm_critic.parameters()) + list(pol.value_net.parameters()))
    opt = th.optim.Adam(params, lr=lr)
    sched = th.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.02)
    rng = np.random.default_rng(seed)
    H = pol.lstm_actor.hidden_size
    for it in range(1, iters + 1):
        idx = rng.choice(len(X), size=batch, replace=False)
        x, m = pad([X[i] for i in idx], np.float32)
        a, _ = pad([A[i] for i in idx], np.int64)
        g, _ = pad([G[i] for i in idx], np.float32)
        z = (th.zeros(1, batch, H), th.zeros(1, batch, H))
        out, _ = pol.lstm_actor(x.unsqueeze(-1), z)
        logits = pol.action_net(pol.mlp_extractor.forward_actor(out))
        w = th.where(a == 1, tap_weight, 1.0) * m  # taps are ~1 in 60 steps
        ce = (th.nn.functional.cross_entropy(logits.reshape(-1, 2), a.reshape(-1),
                                             reduction="none") * w.reshape(-1)).sum() / w.sum()
        outc, _ = pol.lstm_critic(x.unsqueeze(-1), z)
        v = pol.value_net(pol.mlp_extractor.forward_critic(outc)).squeeze(-1)
        vl = (((v - g) ** 2) * m).sum() / m.sum()
        opt.zero_grad()
        (ce + vl).backward()
        th.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0 or it == iters:
            with th.no_grad():
                pred = logits.argmax(-1)
                tp = ((pred == 1) & (a == 1) & (m > 0)).sum().item()
                fp = ((pred == 1) & (a == 0) & (m > 0)).sum().item()
                fn = ((pred == 0) & (a == 1) & (m > 0)).sum().item()
            print(f"  clone iter {it:4d}  CE {ce.item():.4f}  value MSE {vl.item():.4f}  "
                  f"taps: hit {tp} / missed {fn} / extra {fp}", flush=True)


def report(tag, m):
    print(f"[{tag}] reward {m['mean_reward']:+.3f}  mean async {m['mean_asynchrony_ms']:+.1f} ms  "
          f"|async| {m['mean_abs_asynchrony_ms']:.1f} ms  evaluated taps/episode "
          f"{m['taps_per_episode']:.2f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--teacher-episodes", type=int, default=2000)
    p.add_argument("--clone-iters", type=int, default=3000)
    p.add_argument("--tap-weight", type=float, default=10.0, help="CE weight of tap steps")
    p.add_argument("--ppo-steps", type=float, default=3e6)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--out", default="runs/stability_check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    env_cfg = cfg["environment"]
    gamma = float(cfg["rl_algorithm"].get("gamma", 0.99))
    th.manual_seed(args.seed)
    th.set_num_threads(int(cfg.get("torch_threads", 1)))
    os.makedirs(args.out, exist_ok=True)

    model = build_model(cfg, make_vec_env(cfg, args.seed), args.seed,
                        tensorboard_log=os.path.join(args.out, "tb"))
    print(f"Reward {env_cfg['reward_type']}; cloning a scripted on-time tapper from "
          f"{args.teacher_episodes} episodes", flush=True)
    X, A, G = teacher_episodes(env_cfg, args.teacher_episodes, gamma, seed=10_000)
    clone(model, X, A, G, args.clone_iters, batch=64, lr=3e-3, seed=args.seed,
          tap_weight=args.tap_weight)
    m0 = evaluate(model, env_cfg, episodes=args.eval_episodes, seed=777)
    s0 = evaluate(model, env_cfg, episodes=args.eval_episodes, seed=777, deterministic=False)
    report("cloned, deterministic", m0)
    report("cloned, sampled", s0)
    model.save(os.path.join(args.out, "cloned_model"))
    if s0["mean_reward"] < m0["mean_reward"] - 0.5:
        # PPO optimizes the *sampled* policy. If sampling the clone adds many extra taps,
        # PPO will rightly trade them away, so the test says nothing about stability.
        print("INCONCLUSIVE: the cloned policy is not sharp (its sampled return is much "
              "worse than its deterministic return), so PPO would be improving the sampled "
              "policy, not leaving the solution. Re-run with more --clone-iters.")
        sys.exit(2)

    model.learn(total_timesteps=int(args.ppo_steps))
    m1 = evaluate(model, env_cfg, episodes=args.eval_episodes, seed=777)
    s1 = evaluate(model, env_cfg, episodes=args.eval_episodes, seed=777, deterministic=False)
    report(f"after {args.ppo_steps:,.0f} PPO steps, deterministic", m1)
    report(f"after {args.ppo_steps:,.0f} PPO steps, sampled", s1)
    model.save(os.path.join(args.out, "ppo_model"))

    ok = (m1["mean_reward"] >= min(m0["mean_reward"], 0) - 0.1
          and m1["taps_per_episode"] > 0.8 * m0["taps_per_episode"]
          and m1["mean_abs_asynchrony_ms"] < 20)
    print("PASS: PPO keeps the synchronized solution" if ok
          else "FAIL: PPO drifts away from the synchronized solution")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
