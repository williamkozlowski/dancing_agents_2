"""search for learning settings with optuna."""
import argparse
import copy
import json
import os
import sys

import optuna
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import build_model, load_config, make_vec_env  # noqa: E402
from utils import fast_lstm  # noqa: E402
from utils.callbacks import EntropySchedule  # noqa: E402
from utils.evaluation import evaluate  # noqa: E402

fast_lstm.enable()


def suggest(trial, cfg):
    """sample a learning rate, batch size, entropy coefficient, and episode length."""
    cfg = copy.deepcopy(cfg)
    rl, env = cfg["rl_algorithm"], cfg["environment"]
    rl["learning_rate"] = trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True)
    buffer = int(cfg["n_envs"]) * int(rl["n_steps"])
    sizes = [b for b in (256, 512, 1024, 2048, 4096) if buffer % b == 0]
    rl["batch_size"] = trial.suggest_categorical("batch_size", sizes)
    # Entropy coefficient (the start value when config sets an ent_coef_final schedule).
    rl["ent_coef"] = trial.suggest_float("ent_coef", 1e-3, 1e-1, log=True)
    env["sync_beats"] = trial.suggest_int("sync_beats", 6, 20)  # episode horizon (cues)
    return cfg


class Report(BaseCallback):
    def __init__(self, trial, eval_cfg, every, episodes):
        super().__init__(0)
        self.trial, self.eval_cfg, self.every, self.episodes = trial, eval_cfg, every, episodes
        self.next, self.step_idx, self.last = every, 0, None

    def _on_step(self):
        if self.num_timesteps < self.next:
            return True
        self.next += self.every
        m = evaluate(self.model, self.eval_cfg, episodes=self.episodes, seed=4242)
        self.last = m["mean_reward"]
        self.step_idx += 1
        self.trial.report(self.last, self.step_idx)
        print(f"  trial {self.trial.number} @ {self.num_timesteps:,d}: reward {self.last:+.3f} "
              f"taps/ep {m['taps_per_episode']:.1f}", flush=True)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--study", default=None, help="default: tune_<reward_type>")
    p.add_argument("--storage", default="sqlite:///runs/optuna.db")
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--trial-steps", type=float, default=30e6)
    p.add_argument("--eval-every", type=float, default=3e6)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    base = load_config(args.config, args.set)
    name = args.study or "tune_" + base["environment"]["reward_type"].replace("+", "_")
    os.makedirs("runs", exist_ok=True)
    study = optuna.create_study(
        study_name=name, storage=args.storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3))

    if args.show:
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"{len(study.trials)} trials, {len(done)} complete")
        if done:
            print("best reward:", study.best_value)
            print("best params:", json.dumps(study.best_params, indent=2))
        return

    eval_cfg = dict(base["environment"])  # always evaluate on the standard task
    torch.set_num_threads(int(base.get("torch_threads", 1)))

    def objective(trial):
        cfg = suggest(trial, base)
        seed = 1000 + trial.number
        model = build_model(cfg, make_vec_env(cfg, seed), seed)
        cb = [Report(trial, eval_cfg, int(args.eval_every), args.eval_episodes)]
        rl = cfg["rl_algorithm"]
        if rl.get("ent_coef_final") is not None:  # anneal from the suggested start value
            cb.insert(0, EntropySchedule(rl["ent_coef"], rl["ent_coef_final"],
                                         int(rl.get("ent_decay_steps", args.trial_steps))))
        model.learn(total_timesteps=int(args.trial_steps), callback=cb)
        out = os.path.join("runs", name, f"trial_{trial.number}")
        os.makedirs(out, exist_ok=True)
        model.save(os.path.join(out, "model"))
        with open(os.path.join(out, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return evaluate(model, eval_cfg, episodes=100, seed=4242)["mean_reward"]

    study.optimize(objective, n_trials=args.n_trials)
    print("best reward:", study.best_value, "params:", study.best_params)


if __name__ == "__main__":
    main()
