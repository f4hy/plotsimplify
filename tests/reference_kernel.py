"""Pure NumPy reference implementation of the Rust kernel (`scan`, `rdp_keep`).

Same functions and semantics as `plotsimplify._plotsimplify`. The tests swap
it in for the Rust module and require bit-identical results. The search for
each segment's farthest point is vectorized; the heap of segments is a
Python `heapq`, which is cheap because it only holds O(points kept) entries.
"""

import heapq

import numpy as np


def scan(x, y, valid):
    """AND finiteness of x and y into `valid` (in place) and return
    (sorted_ok, first_valid, last_valid, ymin, ymax) over valid points."""
    valid &= np.isfinite(x)
    valid &= np.isfinite(y)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return True, -1, -1, np.inf, -np.inf
    xv = x[idx]
    yv = y[idx]
    ok = not bool(np.any(xv[1:] < xv[:-1]))
    return ok, int(idx[0]), int(idx[-1]), float(yv.min()), float(yv.max())


def _farthest(x, y, s, e, vertical):
    xs = x[s]
    ys = y[s]
    dx = x[e] - xs
    dy = y[e] - ys
    xk = x[s + 1 : e]
    yk = y[s + 1 : e]

    if vertical and dx != 0.0:
        d = np.abs(dy * (xk - xs) - dx * (yk - ys))
        i = int(np.argmax(d))
        return float(d[i]) / abs(dx), s + 1 + i

    if vertical:
        lo = min(ys, y[e])
        hi = max(ys, y[e])
        d = np.maximum(lo - yk, yk - hi)
        i = int(np.argmax(d))
        return float(d[i]), s + 1 + i

    px = xk - xs
    py = yk - ys
    l2 = dx * dx + dy * dy
    d_start = px * px + py * py
    if l2 == 0.0:
        d = d_start
    else:
        t = (px * dx + py * dy) / l2
        qx = px - dx
        qy = py - dy
        c = px * dy - py * dx
        d = np.where(t <= 0.0, d_start, np.where(t >= 1.0, qx * qx + qy * qy, c * c / l2))
    i = int(np.argmax(d))
    return float(np.sqrt(max(d[i], 0.0))), s + 1 + i


def _farthest_multi(x, ys, valid, s, e, vertical):
    """`_farthest` over the rows of ys valid at both s and e: the largest distance,
    ties to the lowest point index, then the first series. None if no row is."""
    best = None
    for y, v in zip(ys, valid):
        if not (v[s] and v[e]):
            continue
        d, k = _farthest(x, y, s, e, vertical)
        if best is None or d > best[0] or (d == best[0] and k < best[1]):
            best = (d, k)
    return best


def _structural(v):
    """Run starts, run ends and the first point of each gap of one mask."""
    n = len(v)
    prev = np.concatenate(([False], v[:-1]))
    nxt = np.concatenate((v[1:], [False]))
    first = np.zeros(n, dtype=np.bool_)
    first[:1] = True
    return np.where(v, ~prev | ~nxt, first | prev)


def rdp_keep(x, y, valid, eps, vertical, max_points, keep):
    """Mark points to keep (in place); return the max error of the result.

    y and valid have shape (k, n): one or more series sharing x, each with its
    own validity mask. Every series' run endpoints and the first point of each
    of its gaps are kept. Between consecutive kept points a series is then
    valid throughout or not drawn at all, so each segment is scored over the
    series valid at both its ends. Segments are split at their farthest point,
    largest distance first, until the largest remaining distance is <= eps or
    `max_points` (if > 0) is reached.
    """
    n = len(x)
    if n == 0:
        return 0.0

    for v in valid:
        keep |= _structural(v)
    kept = np.flatnonzero(keep).tolist()
    count = len(kept)

    heap = []
    for a, b in zip(kept, kept[1:]):
        if b - a >= 2:
            r = _farthest_multi(x, y, valid, a, b, vertical)
            if r is not None:
                heap.append((-r[0], a, b, r[1]))
    heapq.heapify(heap)

    while heap:
        d = -heap[0][0]
        if d <= eps or (max_points > 0 and count >= max_points):
            return d
        _, s, e, k = heapq.heappop(heap)
        keep[k] = True
        count += 1
        for a, b in ((s, k), (k, e)):
            if b - a >= 2:
                r = _farthest_multi(x, y, valid, a, b, vertical)
                if r is not None:
                    heapq.heappush(heap, (-r[0], a, b, r[1]))
    return 0.0
