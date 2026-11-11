"""evaluate agents, save checkpoints, and adjust exploration during learning."""
import csv
import os
import time

from stable_baselines3.common.callbacks import BaseCallback

from utils.evaluation import evaluate


class EvalAndCheckpointCallback(BaseCallback):
    """log evaluation results and save the best and periodic checkpoints."""

    def __init__(self, env_config, total_timesteps, run_dir, eval_freq=200_000,
                 eval_episodes=50, dt=0.01, checkpoint_freq=0, verbose=1):
        super().__init__(verbose)
        self.env_config = env_config
        self.total = int(total_timesteps)
        self.run_dir = run_dir
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.dt = dt
        # Periodic snapshots (checkpoints/step_<N>.zip) so a good agent that later drifts
        # can still be recovered; best_model.zip already keeps the best evaluated one.
        self.checkpoint_freq = int(checkpoint_freq or 0)
        self._next_ckpt = 0
        self._next_eval = 0
        self._start = 0
        self._fractions = {"quarter": 0.25, "half": 0.5}
        self._t0 = time.time()
        self.best_reward = -float("inf")
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, "eval.csv")
        self._csv_header_written = os.path.exists(self.csv_path)

    def _write_row(self, row):
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if not self._csv_header_written:
                w.writeheader()
                self._csv_header_written = True
            w.writerow(row)

    def _evaluate(self):
        m = evaluate(self.model, self.env_config, episodes=self.eval_episodes,
                     seed=10_000_000)
        # The sampled (stochastic) policy is what PPO optimizes; it shows progress that
        # the deterministic P_tap > 0.5 test hides early in training.
        ms = evaluate(self.model, self.env_config, episodes=max(10, self.eval_episodes // 2),
                      seed=20_000_000, deterministic=False)
        m["sampled_reward"] = ms["mean_reward"]
        m["sampled_taps_per_episode"] = ms["taps_per_episode"]
        m["ent_coef"] = float(self.model.ent_coef)
        steps = self.num_timesteps
        row = {"timesteps": steps, "sim_hours": steps * self.dt / 3600.0,
               "wall_min": (time.time() - self._t0) / 60.0, **m}
        if m["taps_per_episode"] > 0 and m["mean_reward"] > self.best_reward:
            self.best_reward = m["mean_reward"]
            self.model.save(os.path.join(self.run_dir, "best_model"))
        self._write_row(row)
        for k, v in m.items():
            self.logger.record(f"eval/{k}", v)
        self.logger.record("eval/sim_hours", row["sim_hours"])
        if self.verbose:
            print(f"[eval] {steps:>11,d} steps ({row['sim_hours']:6.1f} h sim, "
                  f"{row['wall_min']:6.1f} min wall)  reward {m['mean_reward']:+.3f}  "
                  f"async {m['mean_asynchrony_ms']:+6.1f} ms  |async| "
                  f"{m['mean_abs_asynchrony_ms']:5.1f} ms  taps/ep {m['taps_per_episode']:.1f}  "
                  f"| sampled reward {m['sampled_reward']:+.2f} taps/ep "
                  f"{m['sampled_taps_per_episode']:.1f}",
                  flush=True)

    def _on_training_start(self) -> None:
        # A resumed model starts at its saved step count: schedule evaluations and the
        # fractional checkpoints relative to where this run began.
        self._start = self.num_timesteps
        self._next_eval = self.num_timesteps
        self._next_ckpt = self.num_timesteps + self.checkpoint_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._evaluate()
            self._next_eval += self.eval_freq
        if self.checkpoint_freq and self.num_timesteps >= self._next_ckpt:
            d = os.path.join(self.run_dir, "checkpoints")
            os.makedirs(d, exist_ok=True)
            self.model.save(os.path.join(d, f"step_{self.num_timesteps}"))
            self._next_ckpt += self.checkpoint_freq
        for name, frac in list(self._fractions.items()):
            if self.num_timesteps - self._start >= frac * self.total:
                self.model.save(os.path.join(self.run_dir, f"{name}_model"))
                del self._fractions[name]
        return True

    def _on_training_end(self) -> None:
        self._evaluate()


class EntropySchedule(BaseCallback):
    """Linearly anneal PPO's entropy coefficient from ``start`` to ``end`` over
    ``decay_steps`` environment steps, then hold it at ``end``."""

    def __init__(self, start, end, decay_steps):
        super().__init__(0)
        self.start, self.end, self.decay_steps = float(start), float(end), int(decay_steps)
        self._t0 = 0

    def _set(self):
        # Relative to where this run began, so a resumed run anneals from ``start`` too.
        frac = min(1.0, (self.num_timesteps - self._t0) / max(1, self.decay_steps))
        self.model.ent_coef = self.start + frac * (self.end - self.start)

    def _on_training_start(self) -> None:
        self._t0 = self.num_timesteps
        self._set()

    def _on_rollout_start(self) -> None:
        self._set()

    def _on_step(self) -> bool:
        return True
