"""run agents and measure their tap timing."""
import numpy as np

from rl_env.synchronization_env import SynchronizationEnv


def make_env_config(base, **overrides):
    cfg = dict(base)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def run_episode(model, env, deterministic=True, seed=None, options=None, record_hidden=False):
    """run one episode, starting with a fresh recurrent state."""
    obs, _ = env.reset(seed=seed, options=options)
    state = None
    episode_start = np.ones((1,), dtype=bool)
    hidden, actions = [], []
    done = False
    while not done:
        action, state = model.predict(obs, state=state, episode_start=episode_start,
                                      deterministic=deterministic)
        episode_start[:] = False
        if record_hidden:
            hidden.append(np.asarray(state[0]).reshape(-1).copy())  # actor LSTM h_t
        a = int(np.asarray(action).item())
        actions.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        done = terminated or truncated

    result = {
        "ioi": env.ioi, "dt": env.dt,
        "cues": list(env.cue_times),
        "beats": list(env.virtual_cue_times),
        "taps": list(env.tap_buffer),
        "eval_cutoff": env._eval_cutoff,
        "reward": float(env.episode_reward),
        "eval_async": env.asynchronies_ms(),
        "actions": np.asarray(actions, dtype=np.int8),
    }
    if record_hidden:
        result["hidden"] = np.asarray(hidden, dtype=np.float32)
    return result


def tap_for_beat(result, beat_index):
    """Asynchrony (ms) of the tap nearest to a given beat, if one lies within IOI/2."""
    beat = result["beats"][beat_index]
    taps = np.asarray(result["taps"])
    if taps.size == 0:
        return None
    j = int(np.argmin(np.abs(taps - beat)))
    if abs(taps[j] - beat) > result["ioi"] / 2:
        return None
    return float((taps[j] - beat) * result["dt"] * 1000.0)


def evaluate(model, env_config, episodes=100, seed=0, deterministic=True):
    """Mean episode reward and asynchrony over ``episodes`` episodes at random tempi."""
    env = SynchronizationEnv(env_config)
    rewards, ep_means, pooled = [], [], []
    for i in range(episodes):
        r = run_episode(model, env, deterministic=deterministic, seed=seed + i)
        rewards.append(r["reward"])
        if r["eval_async"]:
            ep_means.append(np.mean(r["eval_async"]))
            pooled.extend(r["eval_async"])
    pooled = np.asarray(pooled)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_asynchrony_ms": float(np.mean(ep_means)) if ep_means else float("nan"),
        "mean_abs_asynchrony_ms": float(np.mean(np.abs(pooled))) if pooled.size else float("nan"),
        "tap_sd_ms": float(np.std(pooled)) if pooled.size else float("nan"),
        "taps_per_episode": float(pooled.size / episodes),
    }
