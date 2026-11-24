"""Benchmark plotsimplify against `simplification` (Rust `geo` RDP) and the
pure-NumPy reference kernel from the test suite.

    pip install simplification
    python benchmarks/bench.py

One thread, synthetic sensor data at 1 sample/s, plotted at 1200x400 px.
"""

import sys
import time
from pathlib import Path

import numpy as np

import plotsimplify
from plotsimplify import simd_level, simplify

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import reference_kernel  # noqa: E402

try:
    from simplification.cutil import simplify_coords_idx
except ImportError:
    simplify_coords_idx = None

W, H = 1200, 400


def make_series(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sensor-like series, 1 sample/s: plateaus, ramps, wobbles, light noise, one outage."""
    rng = np.random.default_rng(seed)
    t = np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(1, "s")
    y = np.empty(n)
    i, level = 0, 0.0
    while i < n:
        end = min(n, i + int(rng.integers(50, 2000)))
        kind = rng.random()
        if kind < 0.5:
            y[i:end] = level
        elif kind < 0.85:
            target = level + rng.normal(0, 20)
            y[i:end] = np.linspace(level, target, end - i)
            level = target
        else:
            y[i:end] = level + 5 * np.sin(np.linspace(0, rng.uniform(1, 20), end - i))
        i = end
    y += rng.normal(0, 0.05, n)
    y[n // 3 : n // 3 + n // 200] = np.nan
    return t, y


def best(f, reps):
    out, ts = None, []
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t)
    return out, min(ts) * 1e3


def numpy_ref(t, y, **kw):
    rs = plotsimplify._rs
    plotsimplify._rs = reference_kernel
    try:
        return simplify(t, y, **kw)
    finally:
        plotsimplify._rs = rs


def geo(t, y):
    """`simplification` with both axes scaled to pixels (perpendicular segment
    distance, eps=0.5 px). It can't handle NaN gaps, so they're dropped."""
    ok = np.isfinite(y)
    x = (t[ok] - t[0]).astype(np.float64)
    yy = y[ok]
    coords = np.column_stack(
        [(x - x[0]) * W / (x[-1] - x[0]), (yy - yy.min()) * H / (yy.max() - yy.min())]
    )
    return simplify_coords_idx(np.ascontiguousarray(coords), 0.5)


def main():
    print(f"SIMD path: {simd_level()}")
    print(
        f"{'N':>11} {'case':<30} {'plotsimplify':>13} {'numpy ref':>10} {'simplification':>15} {'kept':>7}"
    )
    for n in (100_000, 1_000_000, 10_000_000):
        t, y = make_series(n)
        reps = 5 if n < 10_000_000 else 2
        cases = [
            ("vertical", dict(metric="vertical")),
            ("perpendicular", dict(metric="perpendicular")),
            ("vertical, max_points=1000", dict(metric="vertical", max_points=1000)),
        ]
        for name, kw in cases:
            kw = dict(kw, width_px=W, height_px=H, eps=0.5)
            r, tr = best(lambda: simplify(t, y, **kw), reps)
            _, tn = best(lambda: numpy_ref(t, y, **kw), 1 if n >= 10_000_000 else 2)
            tg = ""
            if simplify_coords_idx is not None and name == "perpendicular":
                _, g = best(lambda: geo(t, y), 1 if n >= 10_000_000 else 3)
                tg = f"{g:.0f} ms"
            print(f"{n:>11,} {name:<30} {tr:>10.1f} ms {tn:>7.0f} ms {tg:>15} {len(r.indices):>7,}")


if __name__ == "__main__":
    main()
