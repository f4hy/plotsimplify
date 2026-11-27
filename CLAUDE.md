# CLAUDE.md

## Overview

`plotsimplify` does error-bounded Ramer–Douglas–Peucker simplification of time
series for plotting. The Rust kernel is exposed via PyO3/maturin (+ the `numpy`
crate); input handling lives in Python.

- `src/core.rs`: PyO3-free kernels.
  - `scan`: one fused pass that ANDs finiteness into `valid` and returns
    sortedness, the first/last valid index and the y range.
  - `rdp_keep`: max-heap, largest-error-first RDP. Takes k >= 1 y series
    sharing x, each with its own validity mask. It keeps every series' run
    ends and gap starts, then scores each segment by its worst error over
    the series valid at both ends (those are valid throughout it).
  - The 8-lane distance loops (`argmax_cross_*`, `argmax_seg_*`) are compiled
    again with AVX-512/AVX2 `target_feature` and picked at runtime
    (`PLOTSIMPLIFY_BASELINE=1` / `PLOTSIMPLIFY_NO_AVX512=1` force fallbacks).
  - Rust tests compare against textbook recursive RDP.
- `src/lib.rs`: thin numpy-array bindings (`scan`, `rdp_keep`, `simd_level`),
  GIL released. `rdp_keep` takes y and valid as C-contiguous `(k, n)` arrays.
- `python/plotsimplify/__init__.py`: `simplify()` (1-D y) and
  `simplify_many()` (k series), both via `_simplify`. It converts
  datetime64/int x to float offsets from the first valid point (exact in
  int64 first), handles pixel scaling (one shared or per-series y axis) and
  the gap policy, and calls the kernel via the module global `_rs`.
  `gaps="shared"` ANDs all series' finiteness into every row of `valid`;
  `"per_series"` leaves each row to `scan`, then checks x is sorted over the
  union.
- `tests/reference_kernel.py`: pure-NumPy implementation of `scan`/`rdp_keep`.
  Tests monkeypatch `plotsimplify._rs` with it and require **bit-identical**
  results. A subprocess test does the same for the non-SIMD path.
- `docs/downsampling-report.md`: the research that motivated the package
  (mentions Numba/NumPy backends that were dropped when packaging).

## Commands

```sh
cargo test --no-default-features
cargo fmt --check && cargo clippy --no-default-features --all-targets -- -D warnings
pip install -e '.[dev]' && maturin develop --release && pytest
ruff check . && ruff format --check .
pip install simplification && python benchmarks/bench.py
```

## Invariants

- Never enable FMA or reorder floating-point ops in the distance loops: every
  SIMD path and the NumPy reference must round identically. Ties go to the
  lowest index.
- If kernel semantics change, change `tests/reference_kernel.py` the same way.
- Keep `version` in sync between `Cargo.toml` and `pyproject.toml`;
  `abi3-py311` must match `requires-python`.

## Releasing

`release.yml`: `workflow_dispatch` publishes to TestPyPI, and a `v*` tag
publishes to PyPI, both via Trusted Publishing (OIDC). A trusted publisher
must be configured first on pypi.org/test.pypi.org (owner `f4hy`, repo
`plotsimplify`, workflow `release.yml`, environment `pypi`/`testpypi`).
