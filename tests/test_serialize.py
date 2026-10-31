import numpy as np

from robobytes.model import BOS, SEP
from robobytes.serialize import Concat, Stream, quantise


def test_quantise_clip_and_constant():
    x = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    q = quantise(x, np.array([0.0, 5.0]), np.array([1.0, 5.0]))
    assert q[:, 0].tolist() == [0, 255, 255] and q[:, 1].tolist() == [0, 0, 0]


def test_window_matches_symbols():
    s = Stream("toto")
    c = Concat(s, "train")
    ref = np.concatenate([s.symbols(e) for e in c.eps[:3]])
    assert np.array_equal(c.window(0, len(ref)), ref)
    assert np.array_equal(c.window(100, 900), ref[100:1000])
    assert ref[0] == BOS and np.sum(ref[1:s.length[c.eps[0]] * (s.W + 1) + 1] == SEP) == s.length[c.eps[0]]


def test_eval_windows_cover_each_target_once():
    from robobytes.data import eval_windows
    for n, ctx in [(5, 8), (9, 8), (10, 8), (37, 8), (1025, 1024), (3000, 1024)]:
        seen = []
        for s, first in eval_windows(n, ctx):
            seen += [t for t in range(s + 1, min(s + ctx, n - 1) + 1) if t >= first]
        assert seen == list(range(1, n)), (n, ctx)
