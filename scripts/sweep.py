"""run several configurations and seeds in separate processes."""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NO_INIT = ["rl_algorithm.lstm_input_scale=1.0", "rl_algorithm.lstm_forget_bias=null"]
# each preset overrides the base configuration.
VARIANTS = {
    "default": [],
    "const_ent": ["rl_algorithm.normalize_advantage=true", "rl_algorithm.ent_coef=0.01",
                  "rl_algorithm.ent_coef_final=null"],
    "const_ent_no_init": ["rl_algorithm.normalize_advantage=true", "rl_algorithm.ent_coef=0.01",
                          "rl_algorithm.ent_coef_final=null"] + NO_INIT,
    "default_no_init": NO_INIT,
    # anneal entropy to a higher floor.
    "ent_floor": ["rl_algorithm.ent_coef_final=0.03", "rl_algorithm.ent_decay_steps=15000000"],
    # include the previous action.
    "efference": ["environment.efference_input=true"],
    # include tap-landing feedback.
    "tap_feedback": ["environment.tap_feedback=true"],
    "tap_feedback_floor": ["environment.tap_feedback=true", "rl_algorithm.ent_coef_final=0.03",
                           "rl_algorithm.ent_decay_steps=15000000"],
    # anneal entropy when continuing a checkpoint.
    "tap_feedback_anneal": ["environment.tap_feedback=true", "rl_algorithm.ent_coef=0.03",
                            "rl_algorithm.ent_coef_final=0.01",
                            "rl_algorithm.ent_decay_steps=10000000"],
    # Constant entropy at that level; meant for --resume-from a "default" checkpoint.
    "hold": ["rl_algorithm.ent_coef=0.033", "rl_algorithm.ent_coef_final=null"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--sweep", default="sweep_next_interval")
    p.add_argument("--reward-type", default="next+interval")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--only", nargs="+", default=None, choices=list(VARIANTS))
    p.add_argument("--total-timesteps", type=float, default=60e6)
    p.add_argument("--eval-freq", type=int, default=2_000_000)
    p.add_argument("--resume-from", default=None,
                   help="continue from a checkpoint; '{seed}' is replaced by the seed")
    args = p.parse_args()

    for name in args.only or VARIANTS:
        for seed in args.seeds:
            run_dir = os.path.join(ROOT, "runs", args.sweep, name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "train.py"),
                   "--config", args.config, "--seed", str(seed), "--run-dir", run_dir,
                   "--total-timesteps", str(int(args.total_timesteps)),
                   "--set", f"environment.reward_type={args.reward_type}",
                   "--set", f"eval_freq={args.eval_freq}", "--set", "eval_episodes=30"]
            for kv in VARIANTS[name]:
                cmd += ["--set", kv]
            if args.resume_from:
                cmd += ["--resume", args.resume_from.format(seed=seed)]
            log = open(os.path.join(run_dir, "train.log"), "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                    start_new_session=True)
            with open(os.path.join(run_dir, "pid"), "w") as f:
                f.write(str(proc.pid))
            print(f"started {name}/seed_{seed} (pid {proc.pid}) -> {run_dir}")


if __name__ == "__main__":
    main()
