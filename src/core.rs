//! PyO3-free core: the input scan and error-bounded RDP.
//!
//! - `scan`: one fused pass that ANDs finiteness into `valid` and reports
//!   sortedness, the first/last valid index and the y range.
//! - `rdp_keep`: RDP over each run of valid points, splitting segments
//!   largest-error-first (a max-heap) so `max_points` gives the best greedy
//!   budget. Marks kept points and gap markers; returns the max error left.
//!   Takes one or more y series sharing x, each with its own validity mask,
//!   and scores each segment by its worst error over the series drawn across
//!   it, so a single set of kept points bounds every series.
//!
//! The distance loops are written as 8 independent lanes so LLVM vectorises
//! them, and are compiled again for AVX-512/AVX2 and picked at runtime. FMA
//! is never enabled, so every path rounds identically (bit-identical output
//! to the NumPy reference in `tests/reference_kernel.py`).

// ---------------------------------------------------------------- distance

/// Vertical case, dx != 0: argmax of |dy*(xk-xs) - dx*(yk-ys)| over x[a..b].
/// Eight independent lanes so LLVM can vectorise the compare-and-select.
/// Ties go to the lowest index, matching a sequential strict `>` scan.
#[inline(always)]
fn argmax_cross_body(x: &[f64], y: &[f64], xs: f64, ys: f64, dx: f64, dy: f64) -> (f64, usize) {
    const L: usize = 8;
    let n = x.len();
    let mut best = [-1.0f64; L];
    let mut bidx = [0usize; L];
    let chunks = n / L;
    for c in 0..chunks {
        let b = c * L;
        let xc: &[f64; L] = x[b..b + L].try_into().unwrap();
        let yc: &[f64; L] = y[b..b + L].try_into().unwrap();
        for l in 0..L {
            let d = (dy * (xc[l] - xs) - dx * (yc[l] - ys)).abs();
            let gt = d > best[l];
            best[l] = if gt { d } else { best[l] };
            bidx[l] = if gt { b + l } else { bidx[l] };
        }
    }
    let (mut dmax, mut imax) = (-1.0f64, usize::MAX);
    for l in 0..L {
        if best[l] > dmax || (best[l] == dmax && bidx[l] < imax) {
            dmax = best[l];
            imax = bidx[l];
        }
    }
    for i in chunks * L..n {
        let d = (dy * (x[i] - xs) - dx * (y[i] - ys)).abs();
        if d > dmax {
            dmax = d;
            imax = i;
        }
    }
    (dmax, imax)
}

/// Same loop compiled with AVX-512 enabled, where the 8-lane compare-and-select
/// maps onto one masked vector op. (AVX2 alone measured no faster than the baseline
/// build.) FMA is deliberately left off: fusing the multiply-subtract would round
/// differently from the NumPy/Numba backends.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx512bw")]
unsafe fn argmax_cross_avx512(
    x: &[f64],
    y: &[f64],
    xs: f64,
    ys: f64,
    dx: f64,
    dy: f64,
) -> (f64, usize) {
    argmax_cross_body(x, y, xs, ys, dx, dy)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn has_avx512() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    *HAS.get_or_init(|| {
        // PLOTSIMPLIFY_NO_AVX512=1 forces the baseline build: for testing that path,
        // and for measuring what CPUs without AVX-512 get.
        std::env::var_os("PLOTSIMPLIFY_NO_AVX512").is_none()
            && std::env::var_os("PLOTSIMPLIFY_BASELINE").is_none()
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512dq")
            && std::arch::is_x86_feature_detected!("avx512bw")
    })
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static HAS: OnceLock<bool> = OnceLock::new();
    // PLOTSIMPLIFY_BASELINE=1 forces the plain x86-64 build everywhere (testing only).
    *HAS.get_or_init(|| {
        std::env::var_os("PLOTSIMPLIFY_BASELINE").is_none()
            && std::arch::is_x86_feature_detected!("avx2")
    })
}

/// Wheels are built for baseline x86-64, so pick the AVX-512 build at runtime when
/// the CPU has it.
#[inline(always)]
fn argmax_cross(x: &[f64], y: &[f64], xs: f64, ys: f64, dx: f64, dy: f64) -> (f64, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: we just checked the CPU supports these features.
            return unsafe { argmax_cross_avx512(x, y, xs, ys, dx, dy) };
        }
    }
    argmax_cross_body(x, y, xs, ys, dx, dy)
}

/// Perpendicular case, segment length > 0: argmax of squared distance to the
/// segment, clamped to its endpoints. All three cases are computed and one is
/// selected (no branches), in 8 lanes, so it vectorises. Selecting the same
/// expression the scalar code would compute keeps results bit-identical.
#[inline(always)]
fn argmax_seg_body(x: &[f64], y: &[f64], xs: f64, ys: f64, dx: f64, dy: f64) -> (f64, usize) {
    const L: usize = 8;
    let l2 = dx * dx + dy * dy;
    let dist2 = |xk: f64, yk: f64| -> f64 {
        let (px, py) = (xk - xs, yk - ys);
        let t = (px * dx + py * dy) / l2;
        let (qx, qy) = (px - dx, py - dy);
        let c = px * dy - py * dx;
        let d_start = px * px + py * py;
        let d_end = qx * qx + qy * qy;
        let d_mid = c * c / l2;
        if t <= 0.0 {
            d_start
        } else if t >= 1.0 {
            d_end
        } else {
            d_mid
        }
    };
    let n = x.len();
    let mut best = [-1.0f64; L];
    let mut bidx = [0usize; L];
    let chunks = n / L;
    for c in 0..chunks {
        let b = c * L;
        let xc: &[f64; L] = x[b..b + L].try_into().unwrap();
        let yc: &[f64; L] = y[b..b + L].try_into().unwrap();
        for l in 0..L {
            let d = dist2(xc[l], yc[l]);
            let gt = d > best[l];
            best[l] = if gt { d } else { best[l] };
            bidx[l] = if gt { b + l } else { bidx[l] };
        }
    }
    let (mut dmax, mut imax) = (-1.0f64, usize::MAX);
    for l in 0..L {
        if best[l] > dmax || (best[l] == dmax && bidx[l] < imax) {
            dmax = best[l];
            imax = bidx[l];
        }
    }
    for i in chunks * L..n {
        let d = dist2(x[i], y[i]);
        if d > dmax {
            dmax = d;
            imax = i;
        }
    }
    (dmax, imax)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx512dq,avx512bw")]
unsafe fn argmax_seg_avx512(
    x: &[f64],
    y: &[f64],
    xs: f64,
    ys: f64,
    dx: f64,
    dy: f64,
) -> (f64, usize) {
    argmax_seg_body(x, y, xs, ys, dx, dy)
}

// AVX2 tier: measured 28% faster than baseline for this loop (it doesn't help the
// vertical one). FMA left off, as above.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_seg_avx2(
    x: &[f64],
    y: &[f64],
    xs: f64,
    ys: f64,
    dx: f64,
    dy: f64,
) -> (f64, usize) {
    argmax_seg_body(x, y, xs, ys, dx, dy)
}

#[inline(always)]
fn argmax_seg(x: &[f64], y: &[f64], xs: f64, ys: f64, dx: f64, dy: f64) -> (f64, usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            // SAFETY: we just checked the CPU supports these features.
            return unsafe { argmax_seg_avx512(x, y, xs, ys, dx, dy) };
        }
        if has_avx2() {
            // SAFETY: as above.
            return unsafe { argmax_seg_avx2(x, y, xs, ys, dx, dy) };
        }
    }
    argmax_seg_body(x, y, xs, ys, dx, dy)
}

/// Interior point of [s, e] farthest from the drawn segment s-e, and its distance.
#[inline(always)]
fn farthest(x: &[f64], y: &[f64], s: usize, e: usize, vertical: bool) -> (f64, usize) {
    let (xs, ys) = (x[s], y[s]);
    let (dx, dy) = (x[e] - xs, y[e] - ys);
    let (xi, yi) = (&x[s + 1..e], &y[s + 1..e]);

    if vertical && dx != 0.0 {
        let (d, k) = argmax_cross(xi, yi, xs, ys, dx, dy);
        return (d / dx.abs(), s + 1 + k);
    }

    let mut dmax = -1.0f64;
    let mut imax = 0usize;
    if vertical {
        // Endpoints share a timestamp: distance outside [lo, hi].
        let (lo, hi) = (ys.min(y[e]), ys.max(y[e]));
        for (k, &yk) in yi.iter().enumerate() {
            let d = (lo - yk).max(yk - hi);
            if d > dmax {
                dmax = d;
                imax = k;
            }
        }
        return (dmax, s + 1 + imax);
    }

    // Euclidean distance to the segment, clamped to its endpoints. Squared in the loop.
    if dx == 0.0 && dy == 0.0 {
        for (k, (&xk, &yk)) in xi.iter().zip(yi).enumerate() {
            let (px, py) = (xk - xs, yk - ys);
            let d = px * px + py * py;
            if d > dmax {
                dmax = d;
                imax = k;
            }
        }
        return (dmax.max(0.0).sqrt(), s + 1 + imax);
    }
    let (d, k) = argmax_seg(xi, yi, xs, ys, dx, dy);
    (d.max(0.0).sqrt(), s + 1 + k)
}

/// `farthest` over the series drawn across segment s-e: those valid at both
/// ends, which (see `rdp_keep`) are valid at every point in between. Returns the
/// largest distance and its point, or None if no series is drawn there. Ties go
/// to the lowest point index, then the first series.
#[inline(always)]
fn farthest_multi(
    x: &[f64],
    ys: &[&[f64]],
    valids: &[&[bool]],
    s: usize,
    e: usize,
    vertical: bool,
) -> Option<(f64, usize)> {
    // One call site, so `farthest` and its SIMD loops are inlined only once.
    let mut best: Option<(f64, usize)> = None;
    for (y, v) in ys.iter().zip(valids) {
        if !(v[s] && v[e]) {
            continue;
        }
        let (d, k) = farthest(x, y, s, e, vertical);
        best = match best {
            Some((dmax, imax)) if !(d > dmax || (d == dmax && k < imax)) => best,
            _ => Some((d, k)),
        };
    }
    best
}

// ---------------------------------------------------------------- heap

/// Max-heap of segments keyed on the distance of their farthest point.
struct Seg {
    d: f64,
    s: usize,
    e: usize,
    k: usize,
}

struct Heap(Vec<Seg>);

impl Heap {
    fn push(&mut self, seg: Seg) {
        let h = &mut self.0;
        h.push(seg);
        let mut j = h.len() - 1;
        while j > 0 {
            let p = (j - 1) / 2;
            if h[p].d >= h[j].d {
                break;
            }
            h.swap(p, j);
            j = p;
        }
    }

    fn pop(&mut self) -> Seg {
        let h = &mut self.0;
        let top = h.swap_remove(0);
        let n = h.len();
        let mut j = 0;
        loop {
            let mut c = 2 * j + 1;
            if c >= n {
                break;
            }
            if c + 1 < n && h[c + 1].d > h[c].d {
                c += 1;
            }
            if h[c].d <= h[j].d {
                break;
            }
            h.swap(c, j);
            j = c;
        }
        top
    }
}

// ---------------------------------------------------------------- simd level

/// Which SIMD path the distance loops use on this CPU: "avx512", "avx2" or "baseline".
pub fn simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return "avx512";
        }
        if has_avx2() {
            return "avx2"; // perpendicular loop only; vertical uses baseline here
        }
    }
    "baseline"
}

// ---------------------------------------------------------------- kernels

/// RDP over one or more y series sharing `x`. `valids[j]` marks where series j
/// is drawn (finite, see `scan`); it is false in j's gaps.
///
/// First every series' structural points are kept: the start and end of each
/// of its valid runs and the first point of each of its gaps. Between two
/// consecutive kept points a series is then either valid throughout or has no
/// valid point strictly inside (its gap would have put a kept point there), and
/// splitting a segment keeps that true. So a segment is scored only over the
/// series valid at both of its ends. With one series this is plain RDP over
/// each valid run.
pub fn rdp_keep(
    x: &[f64],
    ys: &[&[f64]],
    valids: &[&[bool]],
    eps: f64,
    vertical: bool,
    max_points: i64,
    keep: &mut [bool],
) -> f64 {
    let n = x.len();
    assert!(!ys.is_empty() && ys.len() == valids.len());
    assert!(ys.iter().all(|y| y.len() == n) && valids.iter().all(|v| v.len() == n));
    let mut heap = Heap(Vec::new());
    let mut count: i64 = 0;

    // Run starts and ends, and the first point of each gap. Point 0 is always
    // one or the other. Interior points are done branch-free over three shifted
    // slices, so the pass vectorises.
    if n > 0 {
        keep[0] = true;
    }
    if n >= 2 {
        for v in valids {
            keep[n - 1] |= v[n - 1] || v[n - 2];
            let (prev, cur, next) = (&v[..n - 2], &v[1..n - 1], &v[2..]);
            for (((kp, &p), &c), &nx) in keep[1..n - 1].iter_mut().zip(prev).zip(cur).zip(next) {
                *kp |= (c & !(p & nx)) | (!c & p);
            }
        }
    }
    let mut prev: Option<usize> = None;
    for i in (0..n).filter(|&i| keep[i]) {
        count += 1;
        if let Some(a) = prev {
            if i - a >= 2 {
                if let Some((d, k)) = farthest_multi(x, ys, valids, a, i, vertical) {
                    heap.push(Seg { d, s: a, e: i, k });
                }
            }
        }
        prev = Some(i);
    }

    while let Some(top) = heap.0.first() {
        if top.d <= eps || (max_points > 0 && count >= max_points) {
            return top.d;
        }
        let Seg { s, e, k, .. } = heap.pop();
        keep[k] = true;
        count += 1;
        // Both halves are drawn by the same series as the whole, so neither is None.
        if k - s >= 2 {
            if let Some((d, k1)) = farthest_multi(x, ys, valids, s, k, vertical) {
                heap.push(Seg { d, s, e: k, k: k1 });
            }
        }
        if e - k >= 2 {
            if let Some((d, k2)) = farthest_multi(x, ys, valids, k, e, vertical) {
                heap.push(Seg { d, s: k, e, k: k2 });
            }
        }
    }
    0.0
}

pub fn scan(x: &[f64], y: &[f64], valid: &mut [bool]) -> (bool, i64, i64, f64, f64) {
    // One pass over x and y (the scan is memory-bound, so passes are what cost),
    // in chunks of 8 with independent lanes and no branches, so it vectorises.
    // Within the pass, sortedness is checked between neighbours that are both
    // valid; comparisons across a gap are done afterwards over `valid` only.
    const L: usize = 8;
    let n = x.len();
    let (mut lo, mut hi) = ([f64::INFINITY; L], [f64::NEG_INFINITY; L]);
    let mut unsorted = false;
    let mut n_invalid = 0usize;
    let (mut pv, mut px) = (false, f64::NEG_INFINITY);
    let chunks = n / L;
    for c in 0..chunks {
        let b = c * L;
        let xc: &[f64; L] = x[b..b + L].try_into().unwrap();
        let yc: &[f64; L] = y[b..b + L].try_into().unwrap();
        let vc: &mut [bool; L] = (&mut valid[b..b + L]).try_into().unwrap();
        let mut ok = [false; L];
        for l in 0..L {
            ok[l] = vc[l] & xc[l].is_finite() & yc[l].is_finite();
        }
        for l in 0..L {
            let (pvl, pxl) = if l == 0 {
                (pv, px)
            } else {
                (ok[l - 1], xc[l - 1])
            };
            unsorted |= ok[l] & pvl & (xc[l] < pxl);
            lo[l] = if ok[l] & (yc[l] < lo[l]) {
                yc[l]
            } else {
                lo[l]
            };
            hi[l] = if ok[l] & (yc[l] > hi[l]) {
                yc[l]
            } else {
                hi[l]
            };
            n_invalid += (!ok[l]) as usize;
            vc[l] = ok[l];
        }
        pv = ok[L - 1];
        px = xc[L - 1];
    }
    let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
    for l in 0..L {
        ymin = ymin.min(lo[l]);
        ymax = ymax.max(hi[l]);
    }
    for i in chunks * L..n {
        let ok = valid[i] & x[i].is_finite() & y[i].is_finite();
        unsorted |= ok & pv & (x[i] < px);
        if ok {
            ymin = ymin.min(y[i]);
            ymax = ymax.max(y[i]);
        }
        n_invalid += (!ok) as usize;
        valid[i] = ok;
        pv = ok;
        px = x[i];
    }

    // Across gaps: compare the first valid x after each gap with the last one before it.
    if n_invalid > 0 {
        let mut last_x = f64::NEG_INFINITY;
        let mut in_gap = false;
        for i in 0..n {
            if valid[i] {
                if in_gap {
                    unsorted |= x[i] < last_x;
                    in_gap = false;
                }
                last_x = x[i];
            } else if last_x > f64::NEG_INFINITY {
                in_gap = true;
            }
        }
    }

    let first = valid.iter().position(|&v| v).map_or(-1, |i| i as i64);
    let last = valid.iter().rposition(|&v| v).map_or(-1, |i| i as i64);
    (!unsorted, first, last, ymin, ymax)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG (no dev-dependencies needed).
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn series(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut r = Lcg(seed);
        let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut t, mut v) = (0.0, 0.0);
        for _ in 0..n {
            t += 0.1 + 1.9 * r.f();
            v += r.f() - 0.5;
            x.push(t);
            y.push(v + 3.0 * (t / 20.0).sin());
        }
        (x, y)
    }

    fn vert(x: &[f64], y: &[f64], k: usize, a: usize, b: usize) -> f64 {
        if x[b] == x[a] {
            return (y[a].min(y[b]) - y[k]).max(y[k] - y[a].max(y[b])).max(0.0);
        }
        (y[k] - (y[a] + (y[b] - y[a]) * (x[k] - x[a]) / (x[b] - x[a]))).abs()
    }

    fn seg(x: &[f64], y: &[f64], k: usize, a: usize, b: usize) -> f64 {
        let (dx, dy) = (x[b] - x[a], y[b] - y[a]);
        let l2 = dx * dx + dy * dy;
        let t = if l2 == 0.0 {
            0.0
        } else {
            (((x[k] - x[a]) * dx + (y[k] - y[a]) * dy) / l2).clamp(0.0, 1.0)
        };
        (x[k] - (x[a] + t * dx)).hypot(y[k] - (y[a] + t * dy))
    }

    /// Textbook recursive RDP over one run.
    fn reference(x: &[f64], y: &[f64], eps: f64, vertical: bool) -> Vec<usize> {
        fn rec(
            x: &[f64],
            y: &[f64],
            s: usize,
            e: usize,
            eps: f64,
            vertical: bool,
            keep: &mut Vec<usize>,
        ) {
            if e - s < 2 {
                return;
            }
            let (mut best, mut bi) = (-1.0, 0);
            for k in s + 1..e {
                let d = if vertical {
                    vert(x, y, k, s, e)
                } else {
                    seg(x, y, k, s, e)
                };
                if d > best {
                    best = d;
                    bi = k;
                }
            }
            if best > eps {
                keep.push(bi);
                rec(x, y, s, bi, eps, vertical, keep);
                rec(x, y, bi, e, eps, vertical, keep);
            }
        }
        let mut keep = vec![0, x.len() - 1];
        rec(x, y, 0, x.len() - 1, eps, vertical, &mut keep);
        keep.sort();
        keep
    }

    fn run(x: &[f64], y: &[f64], eps: f64, vertical: bool, max_points: i64) -> (Vec<usize>, f64) {
        let mut valid = vec![true; x.len()];
        scan(x, y, &mut valid);
        let mut keep = vec![false; x.len()];
        let err = rdp_keep(x, &[y], &[&valid], eps, vertical, max_points, &mut keep);
        let idx = keep
            .iter()
            .enumerate()
            .filter(|(_, &k)| k)
            .map(|(i, _)| i)
            .collect();
        (idx, err)
    }

    #[test]
    fn matches_textbook_rdp() {
        for seed in 0..5 {
            let (x, y) = series(400, seed);
            for vertical in [true, false] {
                let (got, _) = run(&x, &y, 1.0, vertical, 0);
                assert_eq!(
                    got,
                    reference(&x, &y, 1.0, vertical),
                    "seed {seed} vertical {vertical}"
                );
            }
        }
    }

    #[test]
    fn error_bound_and_reported_error() {
        let (x, y) = series(5000, 9);
        for vertical in [true, false] {
            for (eps, budget) in [(0.5, 0), (0.0, 150)] {
                let (idx, err) = run(&x, &y, eps, vertical, budget);
                let mut worst: f64 = 0.0;
                for w in idx.windows(2) {
                    for k in w[0] + 1..w[1] {
                        let d = if vertical {
                            vert(&x, &y, k, w[0], w[1])
                        } else {
                            seg(&x, &y, k, w[0], w[1])
                        };
                        worst = worst.max(d);
                    }
                }
                if budget == 0 {
                    assert!(worst <= eps + 1e-9);
                } else {
                    assert_eq!(idx.len(), budget as usize);
                }
                assert!(
                    (worst - err).abs() <= 1e-9 * worst.max(1.0),
                    "{worst} vs {err}"
                );
            }
        }
    }

    #[test]
    fn simd_paths_agree_with_baseline() {
        let (x, y) = series(10_003, 3);
        for (s, e) in [(0, 10_002), (17, 9_000), (5, 30)] {
            let (xs, ys) = (x[s], y[s]);
            let (dx, dy) = (x[e] - xs, y[e] - ys);
            let (xi, yi) = (&x[s + 1..e], &y[s + 1..e]);
            assert_eq!(
                argmax_cross(xi, yi, xs, ys, dx, dy),
                argmax_cross_body(xi, yi, xs, ys, dx, dy)
            );
            assert_eq!(
                argmax_seg(xi, yi, xs, ys, dx, dy),
                argmax_seg_body(xi, yi, xs, ys, dx, dy)
            );
        }
    }

    #[test]
    fn scan_reports_gaps_order_and_range() {
        let x = [0.0, 1.0, 2.0, f64::NAN, 4.0, 5.0];
        let y = [1.0, f64::NAN, -3.0, 0.0, 7.0, 2.0];
        let mut valid = vec![true; 6];
        let (sorted, first, last, lo, hi) = scan(&x, &y, &mut valid);
        assert!(sorted);
        assert_eq!((first, last, lo, hi), (0, 5, -3.0, 7.0));
        assert_eq!(valid, [true, false, true, false, true, true]);

        let x = [0.0, 2.0, f64::NAN, 1.0]; // unsorted across a gap
        let mut valid = vec![true; 4];
        assert!(!scan(&x, &[0.0; 4], &mut valid).0);
    }

    #[test]
    fn gaps_keep_run_ends_and_first_gap_point() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let mut y = vec![0.0; 20];
        for v in &mut y[5..8] {
            *v = f64::NAN;
        }
        y[15] = f64::NAN;
        let (idx, _) = run(&x, &y, 0.5, true, 0);
        assert_eq!(idx, [0, 4, 5, 8, 14, 15, 16, 19]);
    }

    /// Each series' validity comes from its own finiteness (per-series gaps).
    fn run_multi(
        x: &[f64],
        ys: &[&[f64]],
        eps: f64,
        vertical: bool,
        max_points: i64,
    ) -> (Vec<usize>, f64) {
        let valids: Vec<Vec<bool>> = ys
            .iter()
            .map(|y| {
                let mut v = vec![true; x.len()];
                scan(x, y, &mut v);
                v
            })
            .collect();
        let valids: Vec<&[bool]> = valids.iter().map(|v| v.as_slice()).collect();
        let mut keep = vec![false; x.len()];
        let err = rdp_keep(x, ys, &valids, eps, vertical, max_points, &mut keep);
        let idx = (0..x.len()).filter(|&i| keep[i]).collect();
        (idx, err)
    }

    fn worst(x: &[f64], y: &[f64], idx: &[usize], vertical: bool) -> f64 {
        let mut w: f64 = 0.0;
        for p in idx.windows(2) {
            for k in p[0] + 1..p[1] {
                w = w.max(if vertical {
                    vert(x, y, k, p[0], p[1])
                } else {
                    seg(x, y, k, p[0], p[1])
                });
            }
        }
        w
    }

    #[test]
    fn one_series_in_multi_is_unchanged() {
        let (x, y) = series(3000, 4);
        for vertical in [true, false] {
            for (eps, budget) in [(0.5, 0), (0.0, 200)] {
                assert_eq!(
                    run_multi(&x, &[&y, &y], eps, vertical, budget),
                    run(&x, &y, eps, vertical, budget)
                );
            }
        }
    }

    #[test]
    fn band_bounds_every_series() {
        let (x, a) = series(5000, 5);
        let (_, b) = series(5000, 6);
        let lo: Vec<f64> = a.iter().zip(&b).map(|(a, b)| a - 3.0 - b.abs()).collect();
        let hi: Vec<f64> = a.iter().zip(&b).map(|(a, b)| a + 3.0 + b.abs()).collect();
        let ys: [&[f64]; 3] = [&lo, &a, &hi];
        for vertical in [true, false] {
            for (eps, budget) in [(0.5, 0), (0.0, 300)] {
                let (idx, err) = run_multi(&x, &ys, eps, vertical, budget);
                let w = ys
                    .iter()
                    .map(|y| worst(&x, y, &idx, vertical))
                    .fold(0.0, f64::max);
                if budget == 0 {
                    assert!(w <= eps + 1e-9, "{w}");
                } else {
                    assert_eq!(idx.len(), budget as usize);
                }
                assert!((w - err).abs() <= 1e-9 * w.max(1.0), "{w} vs {err}");
            }
        }
    }

    /// Worst error of one series over its own valid runs, which must start and end on kept points.
    fn worst_runs(x: &[f64], y: &[f64], keep: &[usize], vertical: bool) -> f64 {
        let mut w: f64 = 0.0;
        let mut i = 0;
        while i < y.len() {
            if !y[i].is_finite() {
                i += 1;
                continue;
            }
            let s0 = i;
            while i < y.len() && y[i].is_finite() {
                i += 1;
            }
            let ks: Vec<usize> = keep.iter().copied().filter(|&k| k >= s0 && k < i).collect();
            assert_eq!((ks[0], ks[ks.len() - 1]), (s0, i - 1), "run ends kept");
            w = w.max(worst(x, y, &ks, vertical));
        }
        w
    }

    #[test]
    fn per_series_gaps_bound_each_series_on_its_own_runs() {
        let (x, a) = series(4000, 7);
        let (_, mut b) = series(4000, 8);
        let mut c = a.iter().map(|v| v * 2.0).collect::<Vec<_>>();
        for v in &mut b[500..3500] {
            *v = f64::NAN;
        }
        c[100] = f64::NAN;
        c[3999] = f64::NAN;
        let ys: [&[f64]; 3] = [&a, &b, &c];
        for vertical in [true, false] {
            let (idx, err) = run_multi(&x, &ys, 0.5, vertical, 0);
            let w = ys
                .iter()
                .map(|y| worst_runs(&x, y, &idx, vertical))
                .fold(0.0, f64::max);
            assert!(w <= 0.5 + 1e-9, "{w}");
            assert!((w - err).abs() <= 1e-9 * w.max(1.0), "{w} vs {err}");
            // Gap markers of every series are kept.
            for k in [100, 500, 3999] {
                assert!(idx.contains(&k), "{k}");
            }
        }
    }
}
