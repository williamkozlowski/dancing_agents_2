"""speed up recurrent sequence processing when resets occur only at the start."""
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

_original = RecurrentActorCriticPolicy._process_sequence


def _process_sequence(features, lstm_states, episode_starts, lstm):
    n_seq = lstm_states[0].shape[1]
    starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
    if starts.shape[0] > 1 and th.any(starts[1:] != 0.0):
        return _original(features, lstm_states, episode_starts, lstm)
    keep = (1.0 - starts[0]).view(1, n_seq, 1)
    h0, c0 = keep * lstm_states[0], keep * lstm_states[1]
    seq = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
    out, states = lstm(seq, (h0, c0))
    return th.flatten(out.transpose(0, 1), start_dim=0, end_dim=1), states


def self_test(seed=0):
    """True if the fast path reproduces sb3-contrib's own implementation exactly."""
    g = th.Generator().manual_seed(seed)
    lstm = th.nn.LSTM(1, 16)
    n_seq, T = 6, 40
    x = th.rand(n_seq * T, 1, generator=g)
    st = (th.randn(1, n_seq, 16, generator=g), th.randn(1, n_seq, 16, generator=g))
    es = th.zeros(n_seq, T)
    es[::2, 0] = 1.0
    es = es.reshape(-1)
    with th.no_grad():
        a, sa = _original(x, st, es, lstm)
        b, sb = _process_sequence(x, st, es, lstm)
    return (th.allclose(a, b, atol=1e-6) and th.allclose(sa[0], sb[0], atol=1e-6)
            and th.allclose(sa[1], sb[1], atol=1e-6))


def enable():
    """Install the fast path after checking it matches the installed sb3-contrib."""
    if RecurrentActorCriticPolicy._process_sequence is _process_sequence:
        return
    if not self_test():
        raise RuntimeError("utils/fast_lstm.py does not reproduce the installed sb3-contrib's "
                           "_process_sequence; not enabling it (check requirements.txt versions)")
    RecurrentActorCriticPolicy._process_sequence = staticmethod(_process_sequence)
