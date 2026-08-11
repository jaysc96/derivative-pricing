// Turns per-expiry skew curves into 3D surface geometry.
//
// This exists as a pure function, separate from the Plotly-facing component,
// because it is the part that has to be provably right and is the only part
// a test can exercise without a real WebGL context (jsdom has none).
//
// Two decisions carry the same "never draw what wasn't observed" rule the
// 2D skew chart already enforces (LineChart.jsx's maxGapMultiple), extended
// to three dimensions:
//
// 1. **Per-expiry lines break on a strike gap**, exactly like the 2D chart.
//    A quote 20 strikes from its neighbor has nothing observed in between.
// 2. **The mesh uses an alpha-shape, not Delaunay triangulation.** Plotly's
//    mesh3d defaults to `alphahull: -1` (Delaunay across the full convex
//    hull), which would happily bridge the same large gaps a 2D line must
//    not cross. A small positive `alphahull` switches to an alpha-shape,
//    which only connects points closer together than roughly that radius —
//    real gaps stay holes in the mesh instead of being interpolated flat.

//: Matches the 2D skew chart's own default (AnalyticsView.jsx) -- a strike
//: gap wider than this multiple of the curve's typical spacing breaks the
//: line rather than connecting it.
export const GAP_MULTIPLE = 4;

//: The alpha-shape radius, as a multiple of the point cloud's median
//: nearest-neighbor distance (in normalized coordinates -- see below).
//: Tuned empirically against the real archive: 2.5x connects the dense
//: near-the-money region into a continuous sheet while still leaving the
//: same large strike gaps as holes that the 2D chart already excludes.
export const ALPHA_MULTIPLE = 2.5;

/** True median, not `sorted[floor(n/2)]` -- for exactly 2 values that index
 *  picks the larger one outright, which turns a single genuine outlier gap
 *  into its own baseline and defeats the gap-break it's supposed to trigger
 *  (the minimum real-world case: a 3-point skew curve has exactly 2 gaps).
 *  Below 3 values, the minimum is used instead of an average, since with
 *  only one or two samples an average is still pulled toward the outlier;
 *  at 3+ values a real median is safe. */
function median(numbers) {
  if (numbers.length === 0) return 0;
  const sorted = [...numbers].sort((a, b) => a - b);
  if (sorted.length < 3) return sorted[0];
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/** Min/max/range plus a closure mapping a real value to [0, 1]. A
 *  single-value domain (or none) falls back to a range of 1 rather than
 *  dividing by zero; every value then normalizes to 0, which is correct --
 *  there is no spread to represent. */
function normalizer(values) {
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 0;
  const range = max - min || 1;
  return { min, max, range, norm: (v) => (v - min) / range };
}

/** Nearest-neighbor Euclidean distance for each point, brute-force. Point
 *  counts here are in the low hundreds at most (a full multi-expiry skew),
 *  so O(n^2) is a few tens of thousands of comparisons -- not worth a
 *  spatial index for this scale. */
function nearestNeighborDistances(points) {
  return points.map((p, i) => {
    let best = Infinity;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const q = points[j];
      const d = Math.hypot(p.x - q.x, p.y - q.y, p.z - q.z);
      if (d < best) best = d;
    }
    return best;
  });
}

/** `count` evenly spaced ticks across a normalizer's real-value range, each
 *  carrying its normalized position (for Plotly's `tickvals`) and a
 *  formatted label (for `ticktext`). */
function axisTicks(norm, format, count = 5) {
  const ticks = [];
  for (let i = 0; i < count; i++) {
    const real = norm.min + (norm.range * i) / (count - 1);
    ticks.push({ value: norm.norm(real), label: format(real) });
  }
  return ticks;
}

/**
 * @param curves per-expiry skew curves from GET /api/analytics/surface --
 *   `[{ expiry, points: [{ strike, implied_vol }] }]`, any status. Points are
 *   used regardless of the curve's own `insufficient_data` label: that
 *   status means "too few points for a trustworthy 2D line," not "these
 *   points are wrong" -- a thin expiry contributing a few real dots to the
 *   3D mesh is honest in a way a 2-point 2D line claiming a curve shape
 *   would not be.
 * @returns null when no curve has any point, otherwise:
 *   - `lines`: per-expiry `{ expiry, segments }`, each segment a strike-
 *     ascending run of `{ strike, impliedVol, x, y, z }` with no gap wider
 *     than `gapMultiple` times that curve's own median strike spacing.
 *   - `mesh`: `{ x, y, z, alphahull }` -- every point across every curve, one
 *     flat array per axis (mesh3d's expected shape), plus the tuned radius.
 *   - `axisTicks`: `{ x, y, z }`, real-unit tick positions/labels for a
 *     scene built on these normalized coordinates (see module docstring for
 *     why normalization is necessary before `alphahull` means anything).
 */
export function buildSurfaceMesh(curves, { gapMultiple = GAP_MULTIPLE, alphaMultiple = ALPHA_MULTIPLE } = {}) {
  const withPoints = curves.filter((c) => c.points.length > 0);
  if (withPoints.length === 0) return null;

  const xNorm = normalizer(withPoints.flatMap((c) => c.points.map((p) => p.strike)));
  const yNorm = normalizer(withPoints.map((_, i) => i));
  const zNorm = normalizer(withPoints.flatMap((c) => c.points.map((p) => p.implied_vol)));

  const project = (strike, expiryIndex, impliedVol) => ({
    strike,
    impliedVol,
    x: xNorm.norm(strike),
    y: yNorm.norm(expiryIndex),
    z: zNorm.norm(impliedVol),
  });

  const raw = withPoints.flatMap((curve, index) =>
    curve.points.map((p) => project(p.strike, index, p.implied_vol)),
  );

  const lines = withPoints.map((curve, index) => {
    const sorted = [...curve.points].sort((a, b) => a.strike - b.strike);
    const gaps = sorted.slice(1).map((p, i) => p.strike - sorted[i].strike);
    const threshold = median(gaps) * gapMultiple;

    const segments = [[]];
    sorted.forEach((p, i) => {
      if (i > 0 && sorted[i].strike - sorted[i - 1].strike > threshold) segments.push([]);
      segments[segments.length - 1].push(project(p.strike, index, p.implied_vol));
    });

    return { expiry: curve.expiry, segments: segments.filter((s) => s.length > 0) };
  });

  const alphahull = median(nearestNeighborDistances(raw)) * alphaMultiple;

  return {
    lines,
    mesh: { x: raw.map((p) => p.x), y: raw.map((p) => p.y), z: raw.map((p) => p.z), alphahull },
    axisTicks: {
      x: axisTicks(xNorm, (v) => v.toFixed(0)),
      y: withPoints.map((c, i) => ({ value: yNorm.norm(i), label: c.expiry })),
      z: axisTicks(zNorm, (v) => `${(v * 100).toFixed(0)}%`),
    },
  };
}
