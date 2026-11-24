> **Note:** this is the research report that led to `plotsimplify`. It predates the package: the Numba and NumPy backends it mentions were dropped when it was packaged, and only the Rust kernel ships (the NumPy version lives on as the test suite's reference implementation).

# Downsampling Time Series for Plotting in Python

*2026-11-23*

## Summary

For time series sent to a plotting UI, there are two good choices, and which one fits depends on whether you want a guaranteed error or a fixed payload size.

- **Error-bounded, fewest points:** Ramer–Douglas–Peucker (RDP), hand-written in about 30 lines and compiled with Numba, with both axes scaled to pixels and ε ≈ 0.5 px. It simplified 1M points in 27 ms, faster than every library tested.
- **Fixed payload size, noisy data or very large N:** `tsdownsample`'s MinMaxLTTB (or M4 for exact rendering of line charts). 1M points in about 2 ms, and the time doesn't depend on the shape of the data.
- **No compiled dependencies:** RDP in NumPy, with the inner search vectorized. About 4× slower than Numba, still faster than most libraries.
- **Avoid:** the `rdp` package on PyPI. It fails on NumPy 2 and was about 70× slower than plain Python even before that.

Visvalingam–Whyatt mainly earns its place for zoom levels: you compute an importance score for every point once, then filter by threshold. For a single simplification of plateau-and-ramp data, it needed about 7× as many points as RDP for the same error.

## The problem

A chart a few thousand pixels wide can't show millions of samples. Most of them are redundant: they sit on a flat stretch or a straight ramp, or several fall in the same pixel column. The goal is to send the UI only the points that change what the plot looks like.

There are two families of approach:

| Approach | You choose | You get | Algorithms |
| --- | --- | --- | --- |
| Error-bounded line simplification | A tolerance ε | As few points as possible with every dropped point within ε; output size varies | RDP, Visvalingam–Whyatt |
| Fixed-budget downsampling for plots | A point count `n_out` | Exactly `n_out` points chosen to look right at pixel resolution; error varies | LTTB, MinMaxLTTB, MinMax, M4 |

Error-bounded methods suit data with long plateaus and ramps, such as sensor readings. Fixed-budget methods suit noisy, dense data, and any UI that needs a predictable payload size.

## Algorithms

| Algorithm | How it works | Cost | Strength | Weakness |
| --- | --- | --- | --- | --- |
| [RDP](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) | Start with the two endpoints. Find the point farthest from the line between them; if it's farther than ε, keep it and repeat on each half. | Typically O(n log n); O(n²) worst case | Guaranteed max error; fewest points on piecewise-linear data | Keeps every noise point above ε; output size unbounded |
| Visvalingam–Whyatt (VW) | Each point forms a triangle with its neighbours. Repeatedly remove the point with the smallest triangle area until the smallest exceeds ε. | O(n log n) with a heap | Smoother-looking output; gives every point an importance score, which suits zoom levels | ε is an area (x-units × y-units), hard to relate to pixel error |
| LTTB | Split into `n_out` buckets; from each, keep the point forming the largest triangle with the previous kept point and the next bucket's average. | O(n) | Keeps the visual shape well with a fixed budget | Can miss narrow spikes |
| MinMax | Split into `n_out/2` buckets; keep each bucket's min and max. | O(n) | Never loses a spike | Shape between extremes isn't preserved |
| M4 | Keep each bucket's first, last, min and max. | O(n) | With `n_out` = 4 × chart width in pixels, a line chart renders identically to the full data (Jugel et al., VLDB 2014) | 4 points per pixel column |
| MinMaxLTTB | MinMax first narrows the candidates (`minmax_ratio`, default 4), then LTTB runs on them. | O(n) | Nearly identical to LTTB, much faster | Same as LTTB |

RDP builds up from the endpoints, adding the most important points. VW works down from all points, removing the least important. LTTB and its variants make one pass over buckets, which is why they're so fast.

## Speed of each implementation

Hand-written Numba is the fastest exact RDP: 27 ms for 1M points. Times are per call, best of 3 runs, ε = 0.5; sorted by the 1M-point time.

| Implementation | 10k | 100k | 1M | 10M | Notes |
| --- | --- | --- | --- | --- | --- |
| **Numba (hand-written)** | 0.12 ms | 2.2 ms | **27 ms** | 394 ms | Returns indices |
| opencv `approxPolyDP` | 0.48 ms | 4.9 ms | 75 ms | 893 ms | Converts to float32; kept about 8% fewer points |
| shapely 2 (GEOS) `simplify(..., preserve_topology=False)` | 0.38 ms | 7.2 ms | 102 ms | 1.44 s | Returns coordinates, not indices |
| NumPy (hand-written, vectorized inner search) | 0.65 ms | 9.7 ms | 124 ms | 1.85 s | Only needs NumPy |
| `simplification` (Rust `geo`) | 0.94 ms | 18 ms | 237 ms | 3.27 s | Has `simplify_coords_idx` and VW |
| skimage `approximate_polygon` | 5.2 ms | 62 ms | 640 ms | 9.63 s | |
| Pure Python (hand-written, iterative) | 4.9 ms | 103 ms | 1.47 s | — | |
| `rdp` package (`algo="iter"`) | 340 ms | 7.08 s | — | — | Needed a patch to run on NumPy 2 |
| *For reference, not RDP:* `tsdownsample` MinMaxLTTB | 0.06 ms | 0.17 ms | 1.9 ms | 27 ms | Fixed point budget |

"—" means too slow to run. All implementations scale roughly linearly on this data, so the ranking holds at every size. Every exact implementation kept the same points.

## Why the Rust library is slower than Numba

`simplification` is a thin PyO3 wrapper around the Rust [`geo`](https://crates.io/crates/geo) crate, and `geo` 0.31's RDP is written for general geometry rather than speed. Its per-point distance function accounts for about 90% of the gap.

1. **More expensive distance for every point.** It measures distance to the line *segment*, not the infinite line: two divisions, two branches to clamp to the endpoints, and a `hypot()` call that is recomputed for every point though it's the same across a segment. The Numba version does one multiply-subtract and an `abs`, which the compiler can vectorize.
2. **Allocations in the recursion.** Each recursive call returns a new `Vec` that the parent joins, and the input is first copied into a `Vec` of `{index, coord}` structs.

A Numba port of geo's distance function reproduces the slowdown:

| N = 1M, ε = 0.5 | Time (ms) |
| --- | --- |
| Numba, distance to the line | 27.5 |
| Numba, geo-style segment distance with `hypot` | 180.7 |
| `simplification.simplify_coords_idx` | 200.3 |

With a huge ε, which forces a single scan and isolates fixed overhead, `simplification` took 23 ms against Numba's 2.8 ms.

Segment distance isn't wasted work, though. On this test data both kinds of distance kept the same 2,536 points, but they can differ. Take a steep ramp from (0,0) to (1,100) with a spike at (0.5,150): the spike is 50 units from the drawn segment but only 0.5 from the infinite line through it, so infinite-line RDP (including the hand-written versions benchmarked here) can drop real spikes. The fix is to keep segment distance but compute it cheaply: compare squared distances against ε² and drop the per-point `hypot`. Alternatively, use vertical distance, which has no such problem for time series and is the fastest.

## Quality at a similar point budget

RDP gave the smallest error for its point count. The fixed-budget methods show large interpolation error, but they're designed for a different goal, so read the last column with care.

Each method kept about 2,536 points from the same 1M-sample series. Error is the largest vertical gap between the original series and straight lines drawn through the kept points, in y-units.

| Method | Time (ms) | Points kept | Max vertical error |
| --- | --- | --- | --- |
| RDP (Numba), ε = 0.5 | 27.6 | 2,536 | 1.51 |
| VW (`simplification`), area = 5 | 2,273 | 18,134 | 1.59 |
| VW (`simplification`), area = 50 | 2,248 | 3,333 | 5.64 |
| LTTB (`tsdownsample`) | 4.2 | 2,536 | 18.2 |
| MinMaxLTTB (`tsdownsample`) | 2.4 | 2,536 | 23.8 |
| MinMax (`tsdownsample`) | 1.0 | 2,536 | 22.9 |
| M4 (`tsdownsample`) | 0.9 | 2,536 | 24.4 |
| Every n-th point (`tsdownsample`) | <0.1 | 2,536 | 26.6 |

- **VW vs RDP:** to match RDP's error, VW needed about 7× as many points, and `geo`'s VW was about 80× slower.
- **Why the LTTB family scores badly here:** they aim to look right at pixel resolution, not to stay close to the interpolated line. Each bucket here covers about 400 samples, under one pixel on a 1000-px chart. MinMax and M4 are guaranteed to keep every spike; RDP keeps a spike only if it exceeds ε.
- RDP's error (1.51) is larger than ε (0.5) because ε is measured perpendicular to the line. On steep ramps a small perpendicular distance becomes a larger vertical one.

## Pitfalls specific to time series

These matter more than which library you pick.

1. **Scale both axes to pixels before running RDP or VW.** Both measure geometry that mixes x and y units, so with nanosecond timestamps on x, ε means nothing. Map x to the chart's pixel width and y to its pixel height, then ε ≈ 0.5–1 means every dropped point is within about half a pixel of the line.
2. **Consider vertical distance instead of perpendicular.** Since time only moves forward, the vertical gap from the line, |y − interpolated y|, matches what the eye sees and gives a direct bound on vertical error. It's a one-line change in your own implementation; none of the libraries offer it.
3. **RDP's output size has no upper bound.** On noisy data it keeps nearly every point and can slow towards O(n²). Put a cap on it, or fall back to a fixed-budget method.
4. **Zooming.** A simplification made for the full view loses detail when the user zooms in. Re-run it for each visible range (`tsdownsample` is fast enough to do this on every request), precompute a few ε levels, or store VW importance scores and filter by zoom level. [plotly-resampler](https://github.com/predict-idlab/plotly-resampler) does per-viewport resampling for Plotly out of the box.
5. **Return indices, not coordinates.** `simplify_coords_idx`, `tsdownsample` and the Numba version return indices of the kept points. You can then select timestamps (including `datetime64`) and other columns from the original arrays without converting them through float64 (x, y) pairs. `shapely`, `skimage` and `opencv` return only coordinates.

## Trade-offs and recommendations

| Scenario | Use | Why | Trade-off |
| --- | --- | --- | --- |
| Plateau/ramp data, want visually lossless | RDP in Numba, pixel-scaled, ε ≈ 0.5 px | Fewest points with a guaranteed error; 27 ms at 1M points | Output size varies; adds a Numba dependency |
| Same, but no compiled dependencies | RDP in NumPy (vectorized inner search) | Only needs NumPy; 124 ms at 1M points | About 4× slower than Numba |
| Fixed payload size, noisy data, or 10M+ points | `tsdownsample` MinMaxLTTB | About 2 ms at 1M points; time doesn't depend on data shape; returns indices | No error bound; needs sorted x |
| Line chart must look identical to the full data | `tsdownsample` M4 with `n_out` = 4 × chart width in pixels | Matches the full-data render pixel for pixel | 4 points per pixel column, more than RDP on smooth data |
| Zoomable UI | `tsdownsample` on each viewport request, or precomputed VW scores | Fast enough per request; VW scores make any zoom level a simple filter | VW needs a custom implementation to expose scores |
| Already using shapely | `shapely.simplify(line, eps, preserve_topology=False)` | Reasonable speed (102 ms at 1M); no new dependency | Returns coordinates, not indices |
| — | Avoid the `rdp` package | Fails on NumPy 2; 7 s at 100k points | — |

A sensible combination: run RDP at ε ≈ 0.5 px, and if the result is larger than your point budget, fall back to MinMaxLTTB with that budget.

### Reference implementation

This is the Numba version benchmarked above, plus a wrapper that scales to pixel space. On a 1200 × 400 px chart it reduced the 1M-point test series to 1,076 points.

```python
import numpy as np
from numba import njit

@njit(cache=True)
def rdp_indices(x, y, eps):
    """RDP on float64 arrays x, y. Returns indices of kept points."""
    n = len(x)
    keep = np.zeros(n, dtype=np.bool_)
    keep[0] = True
    keep[n - 1] = True
    stack = np.empty((n, 2), dtype=np.int64)
    stack[0, 0], stack[0, 1] = 0, n - 1
    sp = 1
    while sp > 0:
        sp -= 1
        s, e = stack[sp, 0], stack[sp, 1]
        dx, dy = x[e] - x[s], y[e] - y[s]
        thresh = eps * np.sqrt(dx * dx + dy * dy)
        dmax, imax = -1.0, -1
        for i in range(s + 1, e):
            d = abs(dy * (x[i] - x[s]) - dx * (y[i] - y[s]))  # perpendicular distance * segment length
            if d > dmax:
                dmax, imax = d, i
        if imax != -1 and dmax > thresh:
            keep[imax] = True
            stack[sp, 0], stack[sp, 1] = s, imax
            stack[sp + 1, 0], stack[sp + 1, 1] = imax, e
            sp += 2
    return np.flatnonzero(keep)

def simplify_for_plot(t, y, width_px, height_px, eps_px=0.5):
    """Scale to pixel space, then RDP. t may be datetime64 or numeric."""
    t = np.asarray(t).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = (t - t[0]) / max(t[-1] - t[0], 1e-300) * width_px
    ys = (y - y.min()) / max(np.ptp(y), 1e-300) * height_px
    return rdp_indices(x, ys, eps_px)
```

The fixed-budget alternative is one line:

```python
from tsdownsample import MinMaxLTTBDownsampler
idx = MinMaxLTTBDownsampler().downsample(t, y, n_out=2000)  # t must be sorted
```

## Methodology

- **Data:** a synthetic sensor-like series: random-length segments (50–2,000 samples) of flat plateaus (50%), linear ramps (35%) and sine wobbles (15%), plus Gaussian noise with σ = 0.05. x is the sample index.
- **Timing:** the best of 3 runs per implementation (fewer for the slowest), with the Numba function compiled before timing.
- **Environment:** Python 3.12, 8-core Linux machine, November 2026. NumPy 2.5.5, Numba 0.68.1, rdp 0.8, simplification 2.0.0 (Rust `geo` 0.31), shapely 2.2.0, scikit-image 0.26.0, opencv-python-headless 5.1.0, tsdownsample 0.1.5.1.
- **`rdp` package:** patched with a 2D distance function so it would run on NumPy 2.
- **Caveat:** results depend on data shape. Noisier data makes RDP slower and keeps more points, while the fixed-budget methods' speed stays the same.

## Follow-up: our own implementation, three backends

We built `plotsimplify` in this repo with Rust, Numba and NumPy backends behind one API. All three give bit-identical results. It uses the time-series features from the pitfalls section: pixel-scaled ε, a vertical or segment-distance metric, a single-pass `max_points`, NaN gaps and `datetime64` input. The main question was whether Numba's memory cost is worth it. It isn't: Rust is faster and adds almost nothing.

1M points on a 1200×400 px plot, measured in a fresh process per backend (full results in [benchmarks/results.md](benchmarks/results.md)):

| Backend | Vertical | Perpendicular | Memory added by loading it | Startup | Install size |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rust (PyO3, runtime AVX-512/AVX2 dispatch) | 20 ms | 31 ms | +0.3 MB | 1 ms | 0.5 MB |
| Numba | 32 ms | 55 ms | +92 MB | 0.3 s (2.6 s cold) | 201 MB |
| NumPy | 115 ms | 317 ms | +0.4 MB | 1 ms | 0 |

Two things this showed:

- **Most of Rust's lead comes from runtime CPU dispatch.** A portable wheel without it ran the kernel at 17 ms vs 11 ms with it. On a CPU without AVX-512, Rust's lead over Numba shrinks to about 1.25×.
- **Per-call overhead outside the kernel matters at this speed.** Converting `datetime64` and scanning the input take about 5 ms of the 20 ms; the scan is limited by memory bandwidth.
