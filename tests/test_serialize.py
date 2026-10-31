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
