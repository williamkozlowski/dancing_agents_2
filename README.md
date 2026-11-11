# Beat synchronization

A recurrent reinforcement learning agent that learns to tap along with a metronome.
The code includes the environment, agent training, timing tests, and hidden-state analysis.

## Setup

Use Python 3.11 and run these commands from this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Settings live in `configs/config.yaml`. Times in the environment configuration are in
seconds; evaluation reports tap asynchrony in milliseconds. Negative asynchrony means
that the tap landed before the cue.

## Train an agent

```bash
python scripts/train.py --config configs/config.yaml --seed 0
```

The default run uses `next+interval` rewards, 32 environments, and 140 million environment
steps. To choose a shorter run and its output folder:

```bash
python scripts/train.py --config configs/config.yaml --seed 0 \
  --total-timesteps 1000000 --run-dir runs/example/seed_0
```

A short run is useful for checking the setup; it may not learn synchronized tapping.
Use `--set section.key=value` to override settings without editing the config.

### Reward modes

| `environment.reward_type` | Scoring |
| --- | --- |
| `nearest` | Penalizes the distance between a tap and the nearest beat. |
| `next` | Penalizes how far a tap falls before the next beat, with zero penalty on the beat. |
| `nearest+interval` | Adds an inter-tap interval penalty to `nearest`. |
| `next+interval` | Adds an inter-tap interval penalty to `next`. |

The interval penalty increases with the squared difference between the inter-tap interval
and the metronome interval, up to a configurable cap. Missed beats and episodes without
scored taps also receive penalties.

```bash
python scripts/train.py --config configs/config.yaml \
  --set environment.reward_type=nearest --seed 0
python scripts/train.py --config configs/config.yaml \
  --set environment.reward_type=next --seed 0
python scripts/train.py --config configs/config.yaml \
  --set environment.reward_type=nearest+interval --seed 0
```

### Environment options

By default, each episode has 10 audible cues, the first 3 cues are unscored, and the
metronome interval is sampled from 450–750 ms. Each step lasts 10 ms, and taps land
50 ms after initiation. Observations indicate audible cue onsets.

| Setting | Effect |
| --- | --- |
| `min_ioi`, `max_ioi` | Range of metronome intervals; set them equal for a fixed tempo. |
| `exclude_ioi` | Interval range to leave out, such as `[0.5, 0.6]`. |
| `jitter` | Standard deviation of Gaussian cue-time jitter, in seconds. |
| `tap_freeze` | Prevents new taps while another tap is pending. |
| `efference_input` | Adds the previous action to the observation. |
| `sync_beats` | Number of audible cues. |
| `continuation_beats` | Number of silent beats scored after the audible cues. |
| `perturb_beat`, `perturb` | Cue number (starting at 1) and timing shift in seconds; cue 0 disables it. |
| `perturb_mode` | `phase` shifts that cue and all later cues; `event` shifts only that cue. |

All these settings belong under `environment`. Give variants distinct run names to
keep their outputs separate:

```bash
python scripts/train.py --config configs/config.yaml \
  --set environment.jitter=0.04 --set run_name=jitter40
python scripts/train.py --config configs/config.yaml \
  --set environment.tap_freeze=true --set run_name=tap_freeze
python scripts/train.py --config configs/config.yaml \
  --set environment.efference_input=true --set run_name=efference
```

### Run outputs and resuming

Runs are saved to `runs/<reward_mode>/seed_<seed>/`, with `+` replaced by `_`, unless
`run_name` or `--run-dir` is set. Each run writes:

- `config.yaml`: the resolved configuration.
- `eval.csv`: deterministic and sampled evaluation results, including reward and tap timing.
- `best_model.zip`: the best deterministic evaluation checkpoint with scored taps, if one occurs.
- `quarter_model.zip`, `half_model.zip`, `final_model.zip`: checkpoints through the run.
- `checkpoints/step_<N>.zip`: periodic snapshots controlled by `checkpoint_freq`.
- `tb/`: TensorBoard logs.

To continue a checkpoint for another 10 million steps:

```bash
python scripts/train.py --config runs/next_interval/seed_0/config.yaml \
  --resume runs/next_interval/seed_0/final_model.zip \
  --total-timesteps 10000000 --run-dir runs/continued/seed_0
```

Resuming loads the saved model and optimizer. The entropy schedule starts again from
the supplied config's `ent_coef` and runs toward `ent_coef_final`.

## Evaluate an agent

Use the config saved alongside the model, so its observation settings agree with the
agent. Set these paths to the run and checkpoint you want to evaluate:

```bash
R=runs/next_interval/seed_0
M="$R/final_model.zip"
```

| Mode | Command |
| --- | --- |
| Steady metronome, 100 episodes | `python scripts/test_model.py --model "$M" --config "$R/config.yaml"` |
| One trial at a 600 ms interval | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --episodes 1 --ioi 0.6` |
| Jittered cues, 10 ms standard deviation | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --jitter 0.01` |
| Phase shift of 50 ms at cue 6 | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --episodes 1 --perturb-ms 50` |
| Shift only cue 6 | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --episodes 1 --perturb-ms 50 --perturb-mode event` |
| Phase-response curve | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --prc` |
| Ten cues followed by six silent beats | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --continuation` |
| Long trials with 100 audible cues | `python scripts/test_model.py --model "$M" --config "$R/config.yaml" --beats 100` |

`--perturb-beat` selects the shifted cue. Phase-response tests use a 600 ms interval by
default; change it with `--prc-ioi 0.7` for 700 ms. Evaluation uses deterministic actions.

## Analyse hidden states

```bash
python scripts/analyze_hidden.py --model "$M" --config "$R/config.yaml" \
  --out pca_coords.csv --save-hidden hidden_states.npz --plot pca.png
```

This records the actor's recurrent hidden states, fits three principal components, and
reports loop radius, tap clustering, and alternating activity across beats. It runs
1,000 trials across intervals of 450–750 ms by default. Use `--trials` and `--tempi` to
change the number of trials and the interval grid; `--tempi` takes values in milliseconds.
The analysis checks that the recorded states produce the agent's actions and stops if
there are no scored taps or the PCA has no variance. `--allow-no-taps` permits inspecting
an agent that does not tap.

## Search settings and compare runs

Optuna searches the learning rate, batch size, entropy coefficient, and episode length:

```bash
python scripts/tune.py --config configs/config.yaml --n-trials 10 --trial-steps 50e6
python scripts/tune.py --config configs/config.yaml --show
```

Studies are stored in `runs/optuna.db`. Completed trials save a model and config under
`runs/tune_<reward_mode>/trial_<number>/`. `--set` accepts the same overrides as training.

For a preset sweep across seeds:

```bash
python scripts/sweep.py --config configs/config.yaml --only default ent_floor --seeds 0 1
python scripts/summarize_sweep.py runs/sweep_next_interval
```

The sweep starts one background process per configuration and seed. Each run has a
`train.log` and a `pid` file in its output folder. Use `python scripts/sweep.py --help` for the full list of presets.

## Checks and export

Compare scripted tapping strategies under all four reward modes:

```bash
python scripts/check_objective.py --config configs/config.yaml
```

Check whether further learning preserves a cloned on-time tapping strategy:

```bash
python scripts/stability_check.py --config configs/config.yaml
```

The stability check uses a scripted teacher and requires the single metronome input
(`efference_input: false`). It reports pass, fail, or inconclusive; it does not measure
learning from scratch.

To export a selected checkpoint with its config, plain policy weights, and evaluation results:

```bash
python scripts/export_agent.py --model "$M" --config "$R/config.yaml" --out exports/agent
```

## Files

| Path | Contents |
| --- | --- |
| `configs/config.yaml` | Environment, learning, and network settings. |
| `rl_env/synchronization_env.py` | Gymnasium environment and rewards. |
| `scripts/` | Training, evaluation, analysis, search, checks, and export commands. |
| `utils/` | Evaluation helpers, callbacks, and recurrent sequence processing. |
