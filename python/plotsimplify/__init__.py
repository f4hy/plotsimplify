"""plotsimplify: error-bounded time series simplification (RDP) for plotting.

The Ramer-Douglas-Peucker core runs in Rust (SIMD, GIL released); this module
handles input conversion (datetime64, ints, NaN/NaT gaps) and pixel scaling.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple

import numpy as np

from . import _plotsimplify as _rs
from ._plotsimplify import __version__, simd_level

__all__ = ["Simplified", "simplify", "simplify_many", "simd_level", "__version__"]

Metric = Literal["vertical", "perpendicular"]
Gaps = Literal["per_series", "shared"]


class Simplified(NamedTuple):
    """Result of `simplify` and `simplify_many`."""

    indices: np.ndarray
    """int64 indices into the original x/y, sorted."""
    max_error: float
    """Worst distance from a dropped point to the drawn line, in `eps` units.
    For `simplify_many`, the worst over all series."""


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
    y: 1-D values. NaN marks a gap. For several series that share x, use
       `simplify_many`.

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
    y = np.ascontiguousarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be 1-D; for several series sharing x, use simplify_many()")
    if y_range is not None and not _is_pair(y_range):
        raise ValueError("y_range must be a (lo, hi) pair")
    return _simplify(x, y[None, :], eps, width_px, height_px, y_range, metric, max_points, "shared")


def simplify_many(
    x,
    ys,
    *,
    eps: float = 0.5,
    width_px: float | None = None,
    height_px: float | None = None,
    y_range=None,
    metric: Metric = "vertical",
    max_points: int | None = None,
    gaps: Gaps = "per_series",
) -> Simplified:
    """Pick one set of points from which every series in `ys` is drawn within `eps`.

    Use this when several series share x and are sent or drawn together: the
    edges of a band (mu ± sigma, min/max, a confidence interval), stacked
    areas, OHLC, or the columns of a DataFrame. Every series gets the same
    indices, and every series stays within `eps` of its own drawn line.
    Simplifying each series with `simplify` and taking the union of the indices
    doesn't give that: the extra points split the other series' segments, and
    their error can reach 2 * eps.

    ys: the series, as one of
      - a 2-D array of shape (len(x), k), one column per series (like a
        DataFrame's values, or matplotlib's `plot(x, Y)`),
      - a list or tuple of k 1-D arrays,
      - a mapping of name to 1-D array (the order of its values is used).

    gaps: where each series breaks.
      "per_series"  each series has its own gaps (default). A NaN in one series
                    doesn't drop the others' data there.
      "shared"      a point is a gap in every series if any series is NaN
                    there, for drawings that need all series at once, such
                    as a filled band.

    y_range (with width_px/height_px): the y-axis limits.
      None                            one axis, auto-scaled to all the series.
      (lo, hi)                        one axis with fixed limits.
      [r_0, ..., r_{k-1}]             an axis per series, such as twin axes or
                                      small multiples: each entry is (lo, hi),
                                      or None to auto-scale to that series.

    The other arguments are as for `simplify`. If the series are in different
    units and you work in data units (no width_px), scale them first, so that
    one `eps` means the same thing for each: the result is only indices, so
    pick from the unscaled arrays afterwards.
    """
    if gaps not in ("per_series", "shared"):
        raise ValueError(f"gaps must be 'per_series' or 'shared', got {gaps!r}")
    if np.ndim(x) != 1:
        raise ValueError("x must be 1-D")
    y = _as_rows(ys, len(x))
    if y_range is not None and not _is_pair(y_range):
        y_range = list(y_range)
        if len(y_range) != len(y) or not all(r is None or _is_pair(r) for r in y_range):
            raise ValueError("y_range must be (lo, hi), or one (lo, hi) or None per series in ys")
    return _simplify(x, y, eps, width_px, height_px, y_range, metric, max_points, gaps)


def _simplify(x, y, eps, width_px, height_px, y_range, metric, max_points, gaps):
    """y: (k, n) float64 rows. y_range: None, a pair, or a list of k (pair | None)."""
    if metric not in ("vertical", "perpendicular"):
        raise ValueError(f"metric must be 'vertical' or 'perpendicular', got {metric!r}")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    if (width_px is None) != (height_px is None):
        raise ValueError("pass both width_px and height_px, or neither")

    xf, xvalid = _prepare_x(x)
    n = len(xf)
    if y.shape[1] != n:
        raise ValueError("x and y must have the same length")
    k = len(y)

    # One validity row per series. With shared gaps every row is the same mask,
    # so the kernel sees one set of gaps.
    valid = np.repeat(xvalid[None, :], k, axis=0)
    if gaps == "shared" and k > 1:
        valid &= np.isfinite(y).all(axis=0)
    scans = [_rs.scan(xf, y[j], valid[j]) for j in range(k)]
    if gaps == "shared" or k == 1:
        sorted_ok, first, last = scans[0][:3]
    else:
        # Every series is drawn against the same x, so it must be sorted over
        # the points any of them uses, not just within each series.
        union = np.flatnonzero(valid.any(axis=0))
        xu = xf[union]
        sorted_ok = not bool(np.any(xu[1:] < xu[:-1]))
        first, last = (int(union[0]), int(union[-1])) if len(union) else (-1, -1)
    if not sorted_ok:
        raise ValueError("x must be sorted (non-decreasing)")

    vertical = metric == "vertical"
    eps_k = float(eps)
    err_scale = 1.0
    if width_px is not None and first >= 0:
        y0, y1 = _resolve_ranges(y_range, scans)
        sy = [height_px / (b - a) if b > a else 1.0 for a, b in zip(y0, y1)]
        if vertical and all(s == sy[0] for s in sy):
            # Vertical distance in pixels = vertical distance in data units * sy,
            # and x scaling doesn't affect it, so convert eps instead of the arrays.
            eps_k = eps / sy[0]
            err_scale = sy[0]
        elif vertical:
            y = y * np.array(sy)[:, None]
        else:
            x0, x1 = xf[first], xf[last]
            sx = width_px / (x1 - x0) if x1 > x0 else 1.0
            xf = (xf - x0) * sx
            y = (y - np.array(y0)[:, None]) * np.array(sy)[:, None]

    keep = np.zeros(n, dtype=np.bool_)
    err = _rs.rdp_keep(xf, y, valid, eps_k, vertical, max_points or 0, keep)
    return Simplified(np.flatnonzero(keep), float(err) * err_scale)


def _resolve_ranges(y_range, scans):
    """Per-series (y0, y1) lists from y_range and each series' scanned min/max."""
    k = len(scans)
    if isinstance(y_range, list):
        rs = [(s[3], s[4]) if r is None else r for r, s in zip(y_range, scans)]
        return [float(r[0]) for r in rs], [float(r[1]) for r in rs]
    if y_range is None:
        y_range = (min(s[3] for s in scans), max(s[4] for s in scans))
    return [y_range[0]] * k, [y_range[1]] * k


def _is_pair(r):
    """A (lo, hi) pair of numbers, as opposed to a sequence of pairs."""
    try:
        return len(r) == 2 and all(v is not None and np.ndim(v) == 0 for v in r)
    except TypeError:
        return False


def _as_rows(ys, n):
    """ys as a C-contiguous float64 array of shape (k, n): one row per series."""
    if isinstance(ys, Mapping):
        ys = list(ys.values())
    if isinstance(ys, (list, tuple)):
        cols = [np.asarray(c, dtype=np.float64) for c in ys]
        if not cols:
            raise ValueError("ys must hold at least one series")
        if any(c.ndim != 1 for c in cols):
            raise ValueError("each series in ys must be 1-D; for one series, use simplify()")
        if any(len(c) != n for c in cols):
            raise ValueError("each series in ys must have one value per x")
        return np.stack(cols)
    a = np.asarray(ys, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(
            "ys must be a 2-D array of shape (len(x), k), or a list or mapping of 1-D arrays"
        )
    if a.shape[0] != n:
        hint = " (got (k, len(x)); pass ys.T)" if a.shape[1] == n else ""
        raise ValueError(f"a 2-D ys must have shape (len(x), k){hint}")
    if a.shape[1] == 0:
        raise ValueError("ys must hold at least one series")
    return np.ascontiguousarray(a.T)


def _prepare_x(x):
    """x as contiguous float64 plus its validity mask (NaT -> invalid)."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")

    if np.issubdtype(x.dtype, np.datetime64):
        valid = ~np.isnat(x)
        xi = x.view(np.int64)
    elif np.issubdtype(x.dtype, np.integer):
        valid = np.ones(len(x), dtype=np.bool_)
        xi = x.astype(np.int64, copy=False)
    else:
        return np.ascontiguousarray(x, dtype=np.float64), np.ones(len(x), dtype=np.bool_)

    # Subtract the first timestamp while still in int64: ns epochs (~1.7e18)
    # don't fit float64's 53-bit mantissa, but offsets from the start do.
    if len(xi) == 0:
        return np.zeros(0, dtype=np.float64), valid
    first = int(np.argmax(valid)) if valid.any() else 0
    xf = (xi - xi[first]).astype(np.float64)
    return xf, valid
