"""train a recurrent agent to tap along with a metronome."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor  # noqa: E402

from rl_env.synchronization_env import SynchronizationEnv  # noqa: E402
from utils import fast_lstm  # noqa: E402
from utils.callbacks import EntropySchedule, EvalAndCheckpointCallback  # noqa: E402

fast_lstm.enable()  # exact, faster LSTM unrolling during PPO updates


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:  # --set section.key=value
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def make_env_fn(env_config, seed):
    def _init():
        env = SynchronizationEnv(env_config)
        env.reset(seed=seed)
        return env
    return _init


def build_policy_kwargs(cfg):
    pk = dict(cfg.get("policy_kwargs", {}))
    # connect the recurrent output directly to the action and value heads.
    pk.setdefault("net_arch", dict(pi=[], vf=[]))
    if "activation_fn" in pk and isinstance(pk["activation_fn"], str):
        pk["activation_fn"] = getattr(torch.nn, pk["activation_fn"])
    return pk


def make_vec_env(cfg, seed):
    env_cfg = cfg["environment"]
    fns = [make_env_fn(env_cfg, seed * 1000 + i) for i in range(int(cfg.get("n_envs", 16)))]
    vec_cls = SubprocVecEnv if cfg.get("subproc", False) else DummyVecEnv
    return VecMonitor(vec_cls(fns))


def build_model(cfg, env, seed, tensorboard_log=None):
    rl = cfg["rl_algorithm"]
    model = RecurrentPPO(
        policy=rl.get("policy", "MlpLstmPolicy"),
        env=env,
        learning_rate=float(rl.get("learning_rate", 3e-4)),
        n_steps=int(rl.get("n_steps", 1024)),
        batch_size=int(rl.get("batch_size", 1024)),
        n_epochs=int(rl.get("n_epochs", 10)),
        gamma=float(rl.get("gamma", 0.99)),
        gae_lambda=float(rl.get("gae_lambda", 0.95)),
        clip_range=float(rl.get("clip_range", 0.2)),
        ent_coef=float(rl.get("ent_coef", 0.01)),
        vf_coef=float(rl.get("vf_coef", 0.5)),
        max_grad_norm=float(rl.get("max_grad_norm", 0.5)),
        normalize_advantage=bool(rl.get("normalize_advantage", True)),
        target_kl=(float(rl["target_kl"]) if rl.get("target_kl") is not None else None),
        policy_kwargs=build_policy_kwargs(cfg),
        tensorboard_log=tensorboard_log,
        seed=seed,
        device=cfg.get("device", "cpu"),
        verbose=int(cfg.get("verbose", 0)),
    )
    p0 = rl.get("init_tap_prob")
    if p0:
        # Start the perceptron at a small tap probability (~1 tap per beat) instead of
        # P_tap = 0.5 (~400 taps/episode), whose huge initial penalties drive every tap
        # logit far negative and kill exploration before any timing is learned.
        with torch.no_grad():
            b = model.policy.action_net.bias
            b[0] = 0.0
            b[1] = float(torch.logit(torch.tensor(float(p0))))
    init_lstm(model, float(rl.get("lstm_input_scale", 1.0)), rl.get("lstm_forget_bias"))
    return model


def init_lstm(model, input_scale=1.0, forget_bias=None):
    """scale input weights and set the forget gate bias."""
    lstms = [model.policy.lstm_actor]
    if getattr(model.policy, "lstm_critic", None) is not None:
        lstms.append(model.policy.lstm_critic)
    with torch.no_grad():
        for lstm in lstms:
            H = lstm.hidden_size
            for layer in range(lstm.num_layers):
                if layer == 0 and input_scale != 1.0:
                    getattr(lstm, "weight_ih_l0").mul_(input_scale)
                if forget_bias is not None:
                    getattr(lstm, f"bias_ih_l{layer}")[H:2 * H].fill_(float(forget_bias))
                    getattr(lstm, f"bias_hh_l{layer}")[H:2 * H].fill_(0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", default=None, help="output folder (default runs/<name>/seed_<k>)")
    parser.add_argument("--total-timesteps", type=float, default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config entry, e.g. environment.reward_type=next")
    parser.add_argument("--resume", default=None, help="continue training from a saved .zip")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    env_cfg = cfg["environment"]
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    total = int(args.total_timesteps or cfg.get("total_timesteps", 20_000_000))
    name = cfg.get("run_name") or env_cfg.get("reward_type", "agent").replace("+", "_")
    run_dir = args.run_dir or os.path.join("runs", name, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({**cfg, "seed": seed, "total_timesteps": total}, f, sort_keys=False)

    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    env = make_vec_env(cfg, seed)
    if args.resume:
        model = RecurrentPPO.load(args.resume, env=env, device=cfg.get("device", "cpu"),
                                  tensorboard_log=os.path.join(run_dir, "tb"))
        # The saved model keeps its own entropy coefficient; use the config's instead.
        model.ent_coef = float(cfg["rl_algorithm"].get("ent_coef", model.ent_coef))
    else:
        model = build_model(cfg, env, seed, tensorboard_log=os.path.join(run_dir, "tb"))

    callback = [EvalAndCheckpointCallback(
        env_cfg, total, run_dir,
        eval_freq=int(cfg.get("eval_freq", 500_000)),
        eval_episodes=int(cfg.get("eval_episodes", 50)),
        dt=float(env_cfg.get("dt", 0.01)),
        checkpoint_freq=int(cfg.get("checkpoint_freq", 0)),
    )]
    rl = cfg["rl_algorithm"]
    if rl.get("ent_coef_final") is not None:
        callback.insert(0, EntropySchedule(rl.get("ent_coef", 0.01), rl["ent_coef_final"],
                                           int(rl.get("ent_decay_steps", total))))
    print(f"Training {env_cfg.get('reward_type')} | seed {seed} | {total:,d} steps "
          f"({total * float(env_cfg.get('dt', 0.01)) / 3600:.1f} h simulated) -> {run_dir}",
          flush=True)
    model.learn(total_timesteps=total, callback=callback,
                reset_num_timesteps=not bool(args.resume))
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Saved {os.path.join(run_dir, 'final_model.zip')}")


if __name__ == "__main__":
    main()
