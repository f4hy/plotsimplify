"""plotsimplify: error-bounded time series simplification (RDP) for plotting.

The Ramer-Douglas-Peucker core runs in Rust (SIMD, GIL released); this module
handles input conversion (datetime64, ints, NaN/NaT gaps) and pixel scaling.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from . import _plotsimplify as _rs
from ._plotsimplify import __version__, simd_level

__all__ = ["Simplified", "simplify", "simd_level", "__version__"]

Metric = Literal["vertical", "perpendicular"]


class Simplified(NamedTuple):
    """Result of `simplify`."""

    indices: np.ndarray
    """int64 indices into the original x/y, sorted."""
    max_error: float
    """Worst distance from a dropped point to the drawn line, in `eps` units.
    With several y series, the worst over all of them."""


def simplify(
    x,
    y,
    *,
    eps: float = 0.5,
    width_px: float | None = None,
    height_px: float | None = None,
    y_range: tuple[float, float] | None = None,
    metric: Metric = "vertical",
    max_points: int | None = None,
) -> Simplified:
    """Pick the points of a time series needed to draw it within `eps` of the original.

    x: sorted timestamps. numpy datetime64, integer or float.
    y: values. NaN marks a gap. Either one series, or several that share x and
       are drawn together, such as the edges of a band (mu ± sigma, min/max):
       a 2-D array of shape (len(x), k), or a list/tuple of k 1-D arrays.
       Several series get one set of indices, chosen so that every series is
       within `eps` of its own drawn line. A point is a gap if any series is
       NaN there.

    With `width_px` and `height_px`, x and y are scaled to the plot's pixel size
    first, so `eps` is in pixels (0.5 means "every dropped point is within half
    a pixel of the drawn line"). Without them, `eps` is in data units, which is
    only meaningful with metric="vertical".

    y_range: the plot's fixed y-axis limits, if it has them. Defaults to the data's min/max.

    metric:
      "vertical"       distance measured straight up/down from the drawn line (default).
                       Guarantees max |y - drawn line| <= eps.
      "perpendicular"  Euclidean distance to the drawn segment.

    Gaps: points with NaN y (or NaT x) are dropped except the first of each
    gap, which is kept so the plot breaks the line instead of joining across it.

    max_points: keep at most this many points. Segments are split largest error
    first, so you get the best max_points-point approximation the greedy order
    finds; check `max_error` for the bound you actually got. Every gap marker
    and run endpoint is always kept, so data with very many gaps can exceed it.
    """
    if metric not in ("vertical", "perpendicular"):
        raise ValueError(f"metric must be 'vertical' or 'perpendicular', got {metric!r}")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if (width_px is None) != (height_px is None):
        raise ValueError("pass both width_px and height_px, or neither")

    xf, yf, valid = _prepare(x, y)
    n = len(xf)
    if len(yf) > 1:
        valid &= np.isfinite(yf).all(axis=0)
    # With several series `valid` is already final, so every scan sees the same
    # points and returns the same sortedness and first/last; only the y ranges differ.
    scans = [_rs.scan(xf, yj, valid) for yj in yf]
    sorted_ok, first, last, _, _ = scans[0]
    ymin = min(r[3] for r in scans)
    ymax = max(r[4] for r in scans)
    if not sorted_ok:
        raise ValueError("x must be sorted (non-decreasing)")
    if first < 0 or n <= 2:
        return Simplified(np.flatnonzero(valid | _gap_starts(valid)), 0.0)

    vertical = metric == "vertical"
    eps_k = float(eps)
    if width_px is not None:
        y0, y1 = (ymin, ymax) if y_range is None else y_range
        sy = height_px / (y1 - y0) if y1 > y0 else 1.0
        if vertical:
            # Vertical distance in pixels = vertical distance in data units * sy,
            # and x scaling doesn't affect it, so convert eps instead of the arrays.
            eps_k = eps / sy
        else:
            x0, x1 = xf[first], xf[last]
            sx = width_px / (x1 - x0) if x1 > x0 else 1.0
            xf = (xf - x0) * sx
            yf = (yf - y0) * sy

    keep = np.zeros(n, dtype=np.bool_)
    err = _rs.rdp_keep(xf, yf, valid, eps_k, vertical, max_points or 0, keep)
    if width_px is not None and vertical:
        err *= sy
    return Simplified(np.flatnonzero(keep), float(err))


def _gap_starts(valid):
    if len(valid) == 0:
        return valid
    prev = np.concatenate(([True], valid[:-1]))
    return ~valid & prev


def _as_series(y):
    """y as a C-contiguous float64 array of shape (k, n): one row per series."""
    if isinstance(y, (list, tuple)) and len(y) > 0 and np.ndim(y[0]) >= 1:
        cols = [np.asarray(c, dtype=np.float64) for c in y]
        if any(c.ndim != 1 or len(c) != len(cols[0]) for c in cols):
            raise ValueError("each series in y must be a 1-D array of the same length")
        return np.stack(cols)
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        return np.ascontiguousarray(a[None, :])
    if a.ndim == 2:
        if a.shape[1] == 0:
            raise ValueError("y must hold at least one series")
        return np.ascontiguousarray(a.T)
    raise ValueError("y must be 1-D, 2-D of shape (len(x), k), or a sequence of 1-D arrays")


def _prepare(x, y):
    """Convert inputs to contiguous float64 x, (k, n) y and an initial validity mask (NaT -> invalid)."""
    x = np.asarray(x)
    y = _as_series(y)
    if x.ndim != 1 or len(x) != y.shape[1]:
        raise ValueError("x must be 1-D and y must have one value per x")

    if np.issubdtype(x.dtype, np.datetime64):
        valid = ~np.isnat(x)
        xi = x.view(np.int64)
    elif np.issubdtype(x.dtype, np.integer):
        valid = np.ones(len(x), dtype=np.bool_)
        xi = x.astype(np.int64, copy=False)
    else:
        return np.ascontiguousarray(x, dtype=np.float64), y, np.ones(len(x), dtype=np.bool_)

    # Subtract the first timestamp while still in int64: ns epochs (~1.7e18)
    # don't fit float64's 53-bit mantissa, but offsets from the start do.
    first = int(np.argmax(valid)) if valid.any() else 0
    xf = (xi - xi[first]).astype(np.float64)
    return xf, y, valid
