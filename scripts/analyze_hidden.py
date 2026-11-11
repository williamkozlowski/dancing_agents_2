"""record hidden states and compare their trajectories across tempos."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch as th
import yaml
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb3_contrib import RecurrentPPO  # noqa: E402

from rl_env.synchronization_env import SynchronizationEnv  # noqa: E402
from utils.evaluation import make_env_config  # noqa: E402


class AnalysisError(RuntimeError):
    pass


def run_trial(model, env, ioi, seed):
    """One deterministic trial; returns hidden states (T, H) and trial metadata."""
    policy = model.policy
    lstm = policy.lstm_actor
    H = lstm.hidden_size
    h = th.zeros(lstm.num_layers, 1, H)
    c = th.zeros(lstm.num_layers, 1, H)
    state, start = None, np.ones((1,), dtype=bool)

    obs, _ = env.reset(seed=seed, options={"ioi": ioi})
    hidden, done = [], False
    while not done:
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32).view(1, -1)
            feats = policy.extract_features(obs_t)
            feats = feats[0] if isinstance(feats, tuple) else feats
            out, (h, c) = lstm(feats.view(1, 1, -1), (h, c))
            logits = policy.action_net(policy.mlp_extractor.forward_actor(out[0]))
            a_manual = int(th.argmax(logits, dim=-1).item())
        a_pred, state = model.predict(obs, state=state, episode_start=start, deterministic=True)
        start[:] = False
        a_pred = int(np.asarray(a_pred).item())
        if a_pred != a_manual:
            raise AnalysisError(
                f"recorded LSTM state does not reproduce the policy's action at step "
                f"{env.current_step} (manual {a_manual} vs predict {a_pred}); refusing to "
                "analyse states that did not generate the behaviour")
        hidden.append(h[-1, 0].numpy().copy())
        obs, _, terminated, truncated, _ = env.step(a_pred)
        done = terminated or truncated

    hid = np.asarray(hidden, dtype=np.float64)
    T = len(hid)
    cue = np.zeros(T, bool)
    cue[[t for t in env.cue_times if t < T]] = True
    tap = np.zeros(T, bool)
    tap[[t for t in env.tap_buffer if t < T]] = True
    beats = np.asarray(env.virtual_cue_times)
    beat = np.searchsorted(beats, np.arange(T), side="right") - 1
    meta = pd.DataFrame({"step": np.arange(T), "cue": cue, "tap": tap, "beat": beat})
    return hid, meta, len(env.evaluated_taps()), len(beats) - env.entrainment_beats


def collect(model, env_cfg, tempi_ms, n_trials, seed):
    env = SynchronizationEnv(env_cfg)
    H_all, rows, n_taps, n_beats = [], [], 0, 0
    for k in range(n_trials):
        ioi_ms = tempi_ms[k % len(tempi_ms)]
        hid, meta, taps, beats = run_trial(model, env, ioi_ms / 1000.0, seed + k)
        meta.insert(0, "ioi_ms", ioi_ms)
        meta.insert(0, "trial", k)
        H_all.append(hid)
        rows.append(meta)
        n_taps += taps
        n_beats += beats
    return np.concatenate(H_all), pd.concat(rows, ignore_index=True), n_taps, n_beats


def signatures(df, entrainment_beats):
    ev = df[df.beat >= entrainment_beats].copy()
    c1 = ev.groupby("trial").PC1.transform("mean")
    c2 = ev.groupby("trial").PC2.transform("mean")
    ev["radius"] = np.hypot(ev.PC1 - c1, ev.PC2 - c2)
    per_trial = ev.groupby("trial").agg(ioi=("ioi_ms", "first"), radius=("radius", "mean"))
    rho, pval = spearmanr(per_trial.ioi, per_trial.radius)
    by_tempo = per_trial.groupby("ioi").radius.mean()
    print("\nLoop radius in PC1/PC2 by tempo:")
    print("  " + "  ".join(f"{int(i)} ms: {r:.3f}" for i, r in by_tempo.items()))
    print(f"  Spearman(IOI, radius) = {rho:+.2f} (p = {pval:.1e})")

    taps = ev[ev.tap]
    if len(taps) > 1:
        spread = np.hypot(taps.PC1 - taps.PC1.mean(), taps.PC2 - taps.PC2.mean()).mean()
        print(f"Tap clustering: mean tap distance from tap centroid = {spread:.3f} "
              f"vs. mean loop radius {ev.radius.mean():.3f} "
              f"(ratio {spread / ev.radius.mean():.2f}; small = taps in a restricted region)")

    # Period-2 index on beat-locked activity x_b (mean PC value within beat b):
    #   D1 = mean |x_b - x_{b+1}|, D2 = mean |x_b - x_{b+2}|, index = (D1 - D2) / (D1 + D2).
    # Alternation (tick-tock) -> D2 << D1 -> index near +1; steady state -> ~0;
    # slow drift across beats -> D2 ~ 2*D1 -> negative. Drift cannot fake a positive value.
    bm = ev.groupby(["trial", "beat"])[["PC1", "PC2", "PC3"]].mean().reset_index()
    print("Period-2 index per PC, (D1-D2)/(D1+D2) on beat-locked activity "
          "(+1 = alternation between beats, <=0 = none):")
    for pc in ["PC1", "PC2", "PC3"]:
        d1, d2 = [], []
        for _, g in bm.groupby("trial"):
            x = g.sort_values("beat")[pc].to_numpy()
            if len(x) >= 3:
                d1.append(np.abs(np.diff(x)).mean())
                d2.append(np.abs(x[2:] - x[:-2]).mean())
        D1, D2 = float(np.mean(d1)), float(np.mean(d2))
        idx = (D1 - D2) / (D1 + D2) if D1 + D2 > 0 else 0.0
        print(f"  {pc}: {idx:+.2f}")


def plot(df, out_png, entrainment_beats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tempi = sorted(df.ioi_ms.unique())
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(12, 5.5))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    for k, ioi in enumerate(tempi):
        trial = df[df.ioi_ms == ioi].trial.min()
        d = df[(df.trial == trial) & (df.beat >= entrainment_beats - 1)]
        col = cmap(k / max(1, len(tempi) - 1))
        ax3.plot(d.PC1, d.PC2, d.PC3, color=col, lw=0.9, label=f"{ioi:.0f} ms")
        ax2.plot(d.PC1, d.PC2, color=col, lw=0.9)
        t = d[d.tap]
        ax3.scatter(t.PC1, t.PC2, t.PC3, color="k", s=10, depthshade=False)
        ax2.scatter(t.PC1, t.PC2, color="k", s=10, zorder=3)
    ax3.set_xlabel("PC1"); ax3.set_ylabel("PC2"); ax3.set_zlabel("PC3")
    ax3.legend(fontsize=7, loc="upper left")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_title("PC1/PC2 (black = taps)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--tempi", type=float, nargs="+", default=[450, 500, 550, 600, 650, 700, 750],
                   help="test IOIs in ms; trials are spread evenly over them")
    p.add_argument("--beats", type=int, default=None, help="audible cues per trial (default: config)")
    p.add_argument("--out", default="pca_coords.csv")
    p.add_argument("--save-hidden", default=None, help="optional .npz with raw hidden states")
    p.add_argument("--plot", default=None, help="optional PNG of the trajectories")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--allow-no-taps", action="store_true",
                   help="analyse even if the agent never taps (e.g. an untrained agent)")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = make_env_config(cfg["environment"], sync_beats=args.beats)
    model = RecurrentPPO.load(args.model, device="cpu")
    expected_h = model.policy.lstm_actor.hidden_size

    try:
        H, df, n_taps, n_beats = collect(model, env_cfg, args.tempi, args.trials, args.seed)
        if H.ndim != 2 or H.shape[1] != expected_h:
            raise AnalysisError(f"hidden states have shape {H.shape}, expected (T, {expected_h})")
        if not np.isfinite(H).all():
            raise AnalysisError("hidden states contain NaN/inf")
        print(f"Recorded {H.shape[0]:,d} timesteps x {H.shape[1]} hidden units from "
              f"{args.trials} trials at tempi {[int(t) for t in args.tempi]} ms")
        print(f"Evaluated taps: {n_taps} for {n_beats} evaluated beats "
              f"({n_taps / max(1, n_beats):.2f} taps per beat)")
        if n_taps == 0 and not args.allow_no_taps:
            raise AnalysisError("the agent made no evaluated taps in any trial, so there are "
                                "no tapping trajectories to analyse (train longer, or pass "
                                "--allow-no-taps)")
        if n_taps and not 0.5 <= n_taps / n_beats <= 1.5:
            print("WARNING: tap count is far from one per beat; the agent is not synchronizing")

        pca = PCA(n_components=3)
        df[["PC1", "PC2", "PC3"]] = pca.fit_transform(H)
        evr = pca.explained_variance_ratio_
        if not np.isfinite(evr).all() or evr.sum() <= 0:
            raise AnalysisError("PCA is degenerate (hidden states have no variance)")
        print("Explained variance ratio (PC1-3):", np.round(evr, 3), f"total {evr.sum():.3f}")
    except AnalysisError as e:
        sys.exit(f"analyze_hidden.py: ERROR: {e}")

    df.to_csv(args.out, index=False)
    print(f"Saved per-timestep PCs and labels to {args.out}")
    if args.save_hidden:
        np.savez_compressed(args.save_hidden, hidden=H.astype(np.float32),
                            trial=df.trial.values, step=df.step.values, ioi_ms=df.ioi_ms.values)
        print(f"Saved raw hidden states to {args.save_hidden}")
    signatures(df, int(env_cfg.get("entrainment_beats", 3)))
    if args.plot:
        plot(df, args.plot, int(env_cfg.get("entrainment_beats", 3)))


if __name__ == "__main__":
    main()
