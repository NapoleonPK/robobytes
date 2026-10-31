"""Statistics of the pre-registration (sections 2 and 7), on plain arrays. A group is one dataset's paired episodes; pooled
bpb = mean over groups of sum(bits) / sum(bytes). Bootstrap: paired, percentile, 10,000 resamples, RNG seed 0, episodes
resampled within each group; both methods and, for g, numerator and denominator use the same resample.
"""
from __future__ import annotations

import numpy as np

RESAMPLES, SEED = 10_000, 0
G_FLOOR, G_FACTOR, COMPRESSION_FLOOR = 0.02, 1.5, 0.8


def pooled_bpb(groups):
    """groups: list of (bits, bytes), per-episode arrays."""
    return float(np.mean([np.sum(b) / np.sum(n) for b, n in groups]))


def paired_bootstrap(groups, resamples=RESAMPLES, seed=SEED, chunk=500):
    """groups: list of (bits_a, bits_b, bytes). Returns pooled bpb of a and of b per resample."""
    rng = np.random.default_rng(seed)
    a, b = np.zeros(resamples), np.zeros(resamples)
    for bits_a, bits_b, by in groups:
        n = len(by)
        p = np.full(n, 1 / n)
        for i in range(0, resamples, chunk):
            c = rng.multinomial(n, p, size=min(chunk, resamples - i)).astype(np.float64)
            den = c @ by
            a[i:i + len(c)] += c @ bits_a / den
            b[i:i + len(c)] += c @ bits_b / den
    return a / len(groups), b / len(groups)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def excludes_zero(interval):
    return interval[0] > 0 or interval[1] < 0


def compare(groups, resamples=RESAMPLES, seed=SEED):
    """groups: list of (bits_new, bits_base, bytes). diff = new - base (< 0: new better); g = (base - new) / base."""
    new = pooled_bpb([(a, n) for a, _, n in groups])
    base = pooled_bpb([(b, n) for _, b, n in groups])
    ba, bb = paired_bootstrap(groups, resamples, seed)
    return dict(bpb_new=new, bpb_base=base, diff=new - base, diff_ci=ci(ba - bb), g=(base - new) / base,
                g_ci=ci((bb - ba) / bb))


def better(c):
    """H1 and H3: new has lower bpb and the CI of the difference excludes 0."""
    return c["diff"] < 0 and excludes_zero(c["diff_ci"])


def line(g_id):
    return max(G_FACTOR * abs(g_id), G_FLOOR)


def transfer_pass(g_ood, g_ood_ci, g_id):
    """H2 and H2b, one robot: g_OOD > 0 with CI excluding 0, and g_OOD >= max(1.5 |g_ID|, 0.02)."""
    return bool(g_ood > 0 and excludes_zero(g_ood_ci) and g_ood >= line(g_id))


def at_least(flags, k=2):
    return sum(bool(f) for f in flags) >= k


def compression_ok(realised, R_BPE):
    return bool(realised >= COMPRESSION_FLOOR * R_BPE)


def policy_bootstrap(a, b, resamples=RESAMPLES, seed=SEED):
    """H4. a, b: success (seeds x rollouts), same environment seeds per column. Resamples seeds and rollouts, paired.
    Returns difference a - b and its CI."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    S, E = a.shape
    rng = np.random.default_rng(seed)
    d = np.empty(resamples)
    for i in range(resamples):
        ix = np.ix_(rng.integers(0, S, S), rng.integers(0, E, E))
        d[i] = a[ix].mean() - b[ix].mean()
    return dict(diff=float(a.mean() - b.mean()), diff_ci=ci(d))


def uninformative(rates, low=0.05, high=0.95):
    """Every method below 5%, or every method above 95%."""
    rates = np.asarray(rates)
    return bool(np.all(rates < low) or np.all(rates > high))


def flop_multipliers(flops, bpb_ref, bpb_other):
    """Exploratory. For each point of ref's frontier: FLOPs other needs for the same bpb / ref's FLOPs, interpolating other's
    frontier linearly in log FLOPs and log bpb. None when other does not reach that bpb inside the ladder or starts below."""
    fl = np.log(np.asarray(flops, np.float64))
    o = np.log(np.minimum.accumulate(np.asarray(bpb_other, np.float64)))
    out = []
    for f, t in zip(fl, np.log(np.asarray(bpb_ref, np.float64))):
        hit = None
        if o[0] >= t >= o[-1]:
            j = int(np.argmax(o <= t))
            if j == 0:
                hit = fl[0]
            else:
                hit = fl[j - 1] + (t - o[j - 1]) / (o[j] - o[j - 1]) * (fl[j] - fl[j - 1])
        out.append(None if hit is None else float(np.exp(hit - f)))
    return out
