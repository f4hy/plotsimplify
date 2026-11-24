import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import plotsimplify
from plotsimplify import simplify

from . import reference_kernel


# ---------- helpers ----------


def runs_of_valid(y):
    """(start, end) inclusive index pairs of runs of finite y."""
    ok = np.isfinite(y)
    out, i, n = [], 0, len(y)
    while i < n:
        if ok[i]:
            s = i
            while i < n and ok[i]:
                i += 1
            out.append((s, i - 1))
        else:
            i += 1
    return out


def seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def vert_dist(px, py, ax, ay, bx, by):
    if bx == ax:
        return max(min(ay, by) - py, py - max(ay, by), 0.0)
    return abs(py - (ay + (by - ay) * (px - ax) / (bx - ax)))


def max_error(x, y, idx, dist):
    """Largest distance from any original point to the kept segment that spans it."""
    kept = set(idx.tolist())
    worst = 0.0
    for s, e in runs_of_valid(y):
        ks = [k for k in range(s, e + 1) if k in kept]
        assert ks[0] == s and ks[-1] == e, "run endpoints must be kept"
        for a, b in zip(ks, ks[1:]):
            for k in range(a + 1, b):
                worst = max(worst, dist(x[k], y[k], x[a], y[a], x[b], y[b]))
    return worst


def reference_rdp(x, y, eps, dist):
    """Textbook recursive RDP on one run, for comparison."""
    keep = {0, len(x) - 1}

    def rec(s, e):
        if e - s < 2:
            return
        ds = [dist(x[k], y[k], x[s], y[s], x[e], y[e]) for k in range(s + 1, e)]
        i = int(np.argmax(ds))
        if ds[i] > eps:
            k = s + 1 + i
            keep.add(k)
            rec(s, k)
            rec(k, e)

    rec(0, len(x) - 1)
    return np.array(sorted(keep))


def random_series(n, seed):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.uniform(0.1, 2.0, n))
    y = np.cumsum(rng.normal(0, 1, n)) + 3 * np.sin(x / 20)
    return x, y


# ---------- correctness ----------


@pytest.mark.parametrize("metric,dist", [("vertical", vert_dist), ("perpendicular", seg_dist)])
@pytest.mark.parametrize("seed", range(5))
def test_matches_reference(metric, dist, seed):
    x, y = random_series(400, seed)
    got = simplify(x, y, eps=1.0, metric=metric).indices
    np.testing.assert_array_equal(got, reference_rdp(x, y, 1.0, dist))


@pytest.mark.parametrize("metric,dist", [("vertical", vert_dist), ("perpendicular", seg_dist)])
def test_error_bound_holds(metric, dist):
    x, y = random_series(3000, 42)
    for eps in [0.1, 0.5, 2.0]:
        idx = simplify(x, y, eps=eps, metric=metric).indices
        assert max_error(x, y, idx, dist) <= eps + 1e-9


def test_error_bound_in_pixels():
    x, y = random_series(3000, 7)
    w, h, eps = 800, 300, 0.5
    idx = simplify(x, y, eps=eps, width_px=w, height_px=h).indices
    px = (x - x[0]) * w / (x[-1] - x[0])
    py = (y - y.min()) * h / (y.max() - y.min())
    assert max_error(px, py, idx, vert_dist) <= eps + 1e-9


def test_perpendicular_keeps_spike_past_end_of_steep_ramp():
    # Ramp (0,0)->(1,100) with a spike at (0.5,150). The spike is 50 units from
    # the drawn segment but only 0.5 from the infinite line through it, so RDP
    # using infinite-line distance would wrongly drop it.
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 150.0, 100.0])
    for metric in ("perpendicular", "vertical"):
        assert simplify(x, y, eps=1.0, metric=metric).indices.tolist() == [0, 1, 2]


def test_flat_and_linear_collapse_to_endpoints():
    x = np.arange(1000.0)
    assert simplify(x, np.zeros(1000)).indices.tolist() == [0, 999]
    assert simplify(x, 3 * x + 1).indices.tolist() == [0, 999]


def test_duplicate_timestamps():
    # A vertical jump recorded as two samples at the same time, plus a spike there.
    x = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 10.0, 5.0, 5.0])
    idx = simplify(x, y, eps=0.5).indices
    assert max_error(x, y, idx, vert_dist) <= 0.5
    assert 2 in idx  # the spike to 10 must survive


# ---------- gaps, input types ----------


def test_nan_gaps_are_preserved():
    x = np.arange(20.0)
    y = np.zeros(20)
    y[5:8] = np.nan
    y[15] = np.nan
    idx = simplify(x, y).indices.tolist()
    # endpoints of each valid run + first NaN of each gap
    assert idx == [0, 4, 5, 8, 14, 15, 16, 19]


def test_all_nan_and_tiny_inputs():
    assert simplify([], []).indices.tolist() == []
    assert simplify([1.0], [2.0]).indices.tolist() == [0]
    assert simplify([1.0, 2.0], [2.0, 3.0]).indices.tolist() == [0, 1]
    assert simplify(np.arange(5.0), np.full(5, np.nan)).indices.tolist() == [0]


def test_datetime64_matches_seconds():
    x, y = random_series(2000, 3)
    t = np.datetime64("2026-01-01T00:00:00", "ns") + (x * 1e9).astype("timedelta64[ns]")
    a = simplify(t, y, eps=0.5, width_px=1000, height_px=400).indices
    b = simplify(x, y, eps=0.5, width_px=1000, height_px=400).indices
    np.testing.assert_array_equal(a, b)


def test_nat_is_a_gap():
    t = np.arange(10).astype("datetime64[s]")
    t[4] = np.datetime64("NaT", "s")
    idx = simplify(t, np.zeros(10)).indices.tolist()
    assert idx == [0, 3, 4, 5, 9]


def test_unsorted_raises():
    with pytest.raises(ValueError, match="sorted"):
        simplify([0.0, 2.0, 1.0], [0.0, 0.0, 0.0])


def test_bad_args():
    with pytest.raises(ValueError):
        simplify([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        simplify([0.0, 1.0], [0.0, 1.0], width_px=100)
    with pytest.raises(ValueError):
        simplify([0.0, 1.0], [0.0, 1.0], metric="foo")


# ---------- max_points ----------


@pytest.mark.parametrize("metric,dist", [("vertical", vert_dist), ("perpendicular", seg_dist)])
def test_reported_max_error_is_exact(metric, dist):
    x, y = random_series(3000, 11)
    for kw in [dict(eps=0.5), dict(eps=0.0, max_points=150)]:
        r = simplify(x, y, metric=metric, **kw)
        assert r.max_error == pytest.approx(max_error(x, y, r.indices, dist), abs=1e-9)


@pytest.mark.parametrize("budget", [2, 3, 50, 200])
def test_max_points_fills_budget_exactly(budget):
    x, y = random_series(20000, 1)
    assert len(simplify(x, y, eps=0.1).indices) > budget
    r = simplify(x, y, eps=0.1, max_points=budget)
    assert len(r.indices) == budget
    assert r.max_error > 0.1


def test_max_points_larger_than_needed_changes_nothing():
    x, y = random_series(5000, 2)
    a = simplify(x, y, eps=1.0)
    b = simplify(x, y, eps=1.0, max_points=10**9)
    np.testing.assert_array_equal(a.indices, b.indices)
    assert a.max_error == b.max_error <= 1.0


def test_more_budget_never_increases_error_much():
    # Greedy, so not strictly monotonic, but more points should mean lower error overall.
    x, y = random_series(20000, 5)
    errs = [simplify(x, y, eps=0.0, max_points=k).max_error for k in (50, 200, 800, 3200)]
    assert errs == sorted(errs, reverse=True)


# ---------- Rust matches the NumPy reference, bit for bit ----------

CASES = [
    dict(eps=0.5),
    dict(eps=0.0, max_points=777),
    dict(eps=0.2, width_px=900, height_px=300),
]


def gappy_series():
    x, y = random_series(50_000, 9)
    y[1000:1010] = np.nan
    y[20_000] = np.nan
    return x, y


@pytest.mark.parametrize("metric", ["vertical", "perpendicular"])
@pytest.mark.parametrize("kw", CASES)
def test_identical_to_numpy_reference(metric, kw, monkeypatch):
    x, y = gappy_series()
    got = simplify(x, y, metric=metric, **kw)
    monkeypatch.setattr(plotsimplify, "_rs", reference_kernel)
    ref = simplify(x, y, metric=metric, **kw)
    np.testing.assert_array_equal(got.indices, ref.indices)
    assert got.max_error == ref.max_error


def test_baseline_simd_path_is_identical(tmp_path):
    """Force the portable (non-AVX) loops in a subprocess and compare."""
    script = (
        "import numpy as np, sys\n"
        "sys.path.insert(0, %r)\n"
        "from tests.test_simplify import gappy_series, CASES\n"
        "from plotsimplify import simplify, simd_level\n"
        "assert simd_level() == 'baseline', simd_level()\n"
        "x, y = gappy_series()\n"
        "out = [simplify(x, y, metric=m, **kw) for m in ('vertical', 'perpendicular') for kw in CASES]\n"
        "np.save(%r, np.array([r.max_error for r in out]))\n"
        "np.save(%r, np.concatenate([r.indices for r in out]))\n"
    ) % (str(Path(__file__).parent.parent), str(tmp_path / "err.npy"), str(tmp_path / "idx.npy"))
    env = dict(os.environ, PLOTSIMPLIFY_BASELINE="1")
    subprocess.run([sys.executable, "-c", script], check=True, env=env)
    x, y = gappy_series()
    out = [simplify(x, y, metric=m, **kw) for m in ("vertical", "perpendicular") for kw in CASES]
    np.testing.assert_array_equal(
        np.load(tmp_path / "idx.npy"), np.concatenate([r.indices for r in out])
    )
    np.testing.assert_array_equal(np.load(tmp_path / "err.npy"), [r.max_error for r in out])
