# plotsimplify

Shrinks a time series to only the points needed to draw it, using
Ramer–Douglas–Peucker (RDP) with an error bound you set **in pixels**. The
kernel is Rust (SIMD, with runtime CPU dispatch and the GIL released),
exposed to Python via [PyO3](https://pyo3.rs).

```python
import numpy as np
from plotsimplify import simplify

t = np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(1, "s")
y = ...  # float values; NaN = gap

r = simplify(t, y, width_px=1200, height_px=400)   # eps=0.5 px by default
t_small, y_small = t[r.indices], y[r.indices]
r.max_error    # worst error of the result, in pixels (<= eps unless max_points cut it short)
```

On a 1M-point sensor series plotted at 1200×400 px, this keeps about 2,400
points in 20 ms, and every dropped point is within half a pixel of the line
that's drawn.

## Why

Sending or plotting millions of points is slow, and most of them are
invisible. Python already has two kinds of tools for this, and neither fits:

- **RDP libraries aren't built for time series.**
  - The [`rdp`](https://pypi.org/project/rdp/) package is pure Python,
    about 70x slower than a plain loop, and fails on NumPy 2.
  - [`simplification`](https://pypi.org/project/simplification/) (Rust
    `geo`) and shapely are compiled, but they're written for general
    geometry. They have no notion of pixels, datetimes or NaN gaps, and
    `simplification` is 7–9x slower than this package (see below).
- **Fixed-budget downsamplers** such as
  [`tsdownsample`](https://github.com/predict-idlab/tsdownsample) (LTTB,
  MinMax, M4) are very fast and give you exactly `n_out` points, but they
  don't bound the error. They aim to look right at pixel resolution, not to
  stay close to the line. With the same number of points on plateau-and-ramp
  test data, their worst error was 12–16x larger than RDP's.

[`docs/downsampling-report.md`](docs/downsampling-report.md) has the full
comparison of methods and libraries.

## Install

```sh
pip install plotsimplify
```

Prebuilt `abi3` wheels (Linux/macOS/Windows, x86-64 and arm64, Python
3.11+). The only dependency is NumPy.

## Features

- **Error in pixels.** Pass the plot's `width_px`/`height_px` and `eps` is
  in pixels. Without them, `eps` is in data units.
- **Two distance metrics.**
  - `metric="vertical"` (default) measures distance straight up/down from
    the drawn line. It guarantees `max |y - drawn line| <= eps` and is the
    fastest.
  - `metric="perpendicular"` measures Euclidean distance to the drawn
    *segment*. It's clamped to the segment's endpoints, so a spike just
    past the end of a steep ramp isn't dropped. RDP using the infinite line
    would drop it.
- **`max_points`.** Segments are split largest-error-first, so a budget
  gives you the most important points in one pass, and `max_error` reports
  the error you actually got.
- **Several series, one x.** `simplify_many(t, ys)` picks one set of
  indices from which every series is drawn within `eps`. Use it for bands
  (`mu ± sigma`, min/max), stacked areas, OHLC, or a DataFrame's columns.
  Simplifying each series separately and taking the union of the indices
  doesn't give that bound: the extra points split the other series'
  segments, and their error can reach `2 * eps`. The joint pass also keeps
  fewer points.
  - `ys` is a `(len(x), k)` array (one column per series), a list of
    arrays, or a dict of arrays.
  - `gaps="per_series"` (default): each series breaks at its own NaNs, and
    a gap in one series costs the others nothing. `gaps="shared"`: a NaN in
    any series is a gap in all of them, for drawings that need every series
    at once, such as a filled band.
  - `y_range` is one axis for all series (auto or `(lo, hi)`), or a list
    with one `(lo, hi)` or `None` (auto) per series, for twin axes or
    small multiples.

  ```python
  r = simplify_many(t, [lo, hi], width_px=1200, height_px=400, gaps="shared")
  ax.fill_between(t[r.indices], lo[r.indices], hi[r.indices])
  ```
- **Gaps.** Points with NaN `y` or NaT `x` are dropped, except the first
  point of each gap. It's kept so the plot breaks the line instead of
  joining across the gap.
- **Timestamps.** `datetime64` and integer `x` are handled without losing
  precision. `x` must be sorted.
- **Returns indices,** so you can pick timestamps, other columns or metadata
  from your original arrays.
- `y_range=(lo, hi)` if the plot has a fixed y-axis rather than one that
  auto-scales to the data.

## Benchmarks

Intel Xeon W-2123 (AVX-512), one thread, synthetic sensor data at 1 sample/s
with one outage, plotted at 1200×400 px, `eps=0.5` px. Source:
[`benchmarks/bench.py`](./benchmarks/bench.py), raw output in
[`benchmarks/results.txt`](./benchmarks/results.txt).

| N | case | plotsimplify | pure NumPy RDP | `simplification` (Rust geo) | points kept |
|---|---|---:|---:|---:|---:|
| 100k | vertical | **3.8 ms** | 126 ms | – | 7,334 |
| 100k | perpendicular | **4.0 ms** | 115 ms | 38 ms | 1,871 |
| 1M | vertical | **21 ms** | 107 ms | – | 2,386 |
| 1M | perpendicular | **32 ms** | 327 ms | 216 ms | 1,072 |
| 1M | vertical, `max_points=1000` | **19 ms** | 93 ms | – | 1,000 |
| 10M | vertical | **255 ms** | 1,664 ms | – | 14,117 |
| 10M | perpendicular | **345 ms** | 6,560 ms | 3,125 ms | 3,377 |

- `simplification` only has perpendicular distance, so it appears only in
  those rows. It's given the same pixel-scaled coordinates. Its time
  includes that scaling and dropping NaNs, since it can't handle gaps.
- The "pure NumPy RDP" column is the test suite's reference implementation
  (`tests/reference_kernel.py`): the same algorithm with the inner search
  vectorized. It's a good stand-in for what you'd get writing RDP in NumPy
  yourself.

## How it's fast

- **The farthest-point search uses 8 independent lanes,** so the compiler
  can turn it into SIMD. Ties go to the lowest index.
- **Runtime CPU dispatch.** Wheels are built for baseline x86-64, so the hot
  loops are also compiled for AVX-512 and AVX2 and chosen at startup.
  `plotsimplify.simd_level()` reports which path is in use. Without dispatch,
  a portable wheel lost about a third of its speed. arm64 builds use the
  generic loops with NEON; that path isn't benchmarked yet.
- **FMA stays off.** Fused multiply-add would round differently, so every
  SIMD path gives bit-identical output, and the output matches a plain NumPy
  implementation. The tests check both.
- **Perpendicular distance is computed branch-free:** all three cases are
  computed and one is selected, so it vectorizes too.
- **The input scan is one fused pass,** because it's memory-bound: the
  number of passes over the data is what costs.
- **Pixel scaling is free for `vertical`.** Vertical pixel distance is just
  data distance times a constant, so `eps` is converted instead of the
  arrays.

## Using it in a web service

- **Simplify only the visible range.** Slice to the zoom window with
  `np.searchsorted` first, then simplify. Zooming in brings back detail.
- **Cap the payload.** Zoomed in, sensor noise can be several pixels tall,
  and RDP keeps noise. `max_points=4 * width_px` is a reasonable cap.
- **Use a sync endpoint.** In FastAPI, a plain `def` endpoint runs in the
  thread pool, and `simplify` releases the GIL, so concurrent requests run
  in parallel.
- **Gaps in JSON.** Send `y[r.indices]` with NaN gap markers as `null`;
  plotting libraries draw a break there.

## Development

```sh
cargo test --no-default-features
pip install -e '.[dev]' && maturin develop --release
pytest    # includes bit-identical checks vs the NumPy reference and the non-SIMD path
pip install simplification && python benchmarks/bench.py
```

## License

MIT
