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


def rdp_keep(x, y, valid, eps, vertical, max_points, keep):
    """Mark points to keep (in place); return the max error of the result.

    Every gap's first point and every run's endpoints are kept; then segments
    are split at their farthest point, largest distance first, until the
    largest remaining distance is <= eps or `max_points` (if > 0) is reached.
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Runs of valid points and the first index of each gap, found with array ops.
    v = valid.astype(np.int8)
    edges = np.diff(np.concatenate(([0], v, [0])))
    run_starts = np.flatnonzero(edges == 1)
    run_ends = np.flatnonzero(edges == -1) - 1
    gap_starts = np.flatnonzero(edges[:-1] == -1)  # index right after a run ends
    if not valid[0]:
        gap_starts = np.concatenate(([0], gap_starts))
    gap_starts = gap_starts[gap_starts < n]

    keep[gap_starts] = True
    keep[run_starts] = True
    keep[run_ends] = True
    count = int(np.count_nonzero(keep))

    heap = []
    for s0, e0 in zip(run_starts.tolist(), run_ends.tolist()):
        if e0 - s0 >= 2:
            d, k = _farthest(x, y, s0, e0, vertical)
            heap.append((-d, s0, e0, k))
    heapq.heapify(heap)

    while heap:
        d = -heap[0][0]
        if d <= eps or (max_points > 0 and count >= max_points):
            return d
        _, s, e, k = heapq.heappop(heap)
        keep[k] = True
        count += 1
        if k - s >= 2:
            d1, k1 = _farthest(x, y, s, k, vertical)
            heapq.heappush(heap, (-d1, s, k, k1))
        if e - k >= 2:
            d2, k2 = _farthest(x, y, k, e, vertical)
            heapq.heappush(heap, (-d2, k, e, k2))
    return 0.0
