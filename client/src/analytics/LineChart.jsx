// A small SVG line chart with real axes.
//
// Hand-rolled rather than pulled from a charting library: the three views
// here are the same shape (one or more series of x/y points on a linear
// scale) and need no zoom, brushing, or animation, so a dependency would
// cost bundle size and a CDN-or-vendored decision for markup we can read.
//
// `series` is `[{ name, color, points: [{ x, y }] }]`. A `null` y breaks the
// line into segments rather than plotting zero -- the implied-vs-realized
// view has days where only one of its two series has a value, and drawing
// those as a dive to the axis would invent a reading the archive never took.

const MARGIN = { top: 12, right: 14, bottom: 34, left: 52 };

/** Round a domain out to readable tick values (1/2/5 x 10^n steps). */
function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const rawStep = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const step = (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  const first = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = first; v <= max + step * 1e-9; v += step) {
    // Re-round to kill float drift like 0.30000000000000004 in tick labels.
    ticks.push(Number(v.toFixed(10)));
  }
  return ticks;
}

/** Median gap between consecutive x-values in an ascending, unique-x series.
 *
 * A 3-point series has exactly 2 gaps -- the minimum size a real skew curve
 * ever has -- and `sorted[floor(2/2)]` picks index 1, the *larger* of the
 * two, not their average. When one of those two gaps is the genuine
 * far-out-of-the-money outlier this function exists to detect, that outlier
 * becomes its own baseline and the break it should trigger never fires.
 * Below 3 gaps, the minimum (not an average pulled toward the outlier) is
 * the only value immune to that -- above 3, a real median is safe since one
 * outlier can no longer dominate either middle element. */
function medianGap(points) {
  if (points.length < 2) return Infinity;
  const gaps = [];
  for (let i = 1; i < points.length; i++) gaps.push(points[i].x - points[i - 1].x);
  gaps.sort((a, b) => a - b);
  if (gaps.length < 3) return gaps[0];
  const mid = Math.floor(gaps.length / 2);
  return gaps.length % 2 === 0 ? (gaps[mid - 1] + gaps[mid]) / 2 : gaps[mid];
}

export default function LineChart({
  series,
  width = 560,
  height = 240,
  xTicks,
  xFormat = (v) => String(v),
  yFormat = (v) => v.toFixed(2),
  xLabel,
  yLabel,
  ariaLabel,
  // A gap between consecutive points more than this multiple of the series'
  // own median spacing breaks the line rather than connecting it. A quote
  // sitting 20 strikes from its neighbor (deep out-of-the-money wings are
  // thin) has no observations in between -- a straight line across that gap
  // draws a smooth transition the archive never actually saw. `Infinity`
  // (the default) never breaks, which keeps every other chart unchanged.
  maxGapMultiple = Infinity,
}) {
  const innerWidth = width - MARGIN.left - MARGIN.right;
  const innerHeight = height - MARGIN.top - MARGIN.bottom;

  const valid = series.flatMap((s) => s.points.filter((p) => p.y !== null && p.y !== undefined));
  if (valid.length === 0) return null;

  const xs = valid.map((p) => p.x);
  const ys = valid.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yLo = Math.min(...ys);
  const yHi = Math.max(...ys);
  // A flat series still needs a visible band, hence the fallbacks.
  const pad = (yHi - yLo) * 0.12 || Math.abs(yHi) * 0.12 || 0.01;
  const yMin = yLo - pad;
  const yMax = yHi + pad;

  const scaleX = (x) => ((x - xMin) / (xMax - xMin || 1)) * innerWidth;
  const scaleY = (y) => innerHeight - ((y - yMin) / (yMax - yMin || 1)) * innerHeight;

  const yTickValues = niceTicks(yMin, yMax, 4);
  const xTickValues =
    xTicks && xTicks.length > 0
      ? xTicks
      : niceTicks(xMin, xMax, 5).map((v) => ({ value: v, label: xFormat(v) }));

  function pathFor(points) {
    const gapThreshold = medianGap(points) * maxGapMultiple;
    const segments = [[]];
    let previous = null;
    for (const p of points) {
      if (p.y === null || p.y === undefined) {
        if (segments[segments.length - 1].length) segments.push([]);
        previous = null;
        continue;
      }
      if (previous !== null && p.x - previous > gapThreshold) {
        segments.push([]);
      }
      segments[segments.length - 1].push(p);
      previous = p.x;
    }
    return segments
      .filter((segment) => segment.length > 0)
      .map((segment) =>
        segment.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" "),
      )
      .join(" ");
  }

  return (
    <svg
      className="chart"
      role="img"
      aria-label={ariaLabel}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        {yTickValues.map((value) => (
          <g key={`y${value}`} transform={`translate(0, ${scaleY(value)})`}>
            <line className="chart-grid-line" x1="0" x2={innerWidth} y1="0" y2="0" />
            <text className="chart-tick-label" x="-8" y="0" textAnchor="end" dominantBaseline="middle">
              {yFormat(value)}
            </text>
          </g>
        ))}

        {xTickValues.map((tick) => (
          <text
            key={`x${tick.value}-${tick.label}`}
            className="chart-tick-label"
            x={scaleX(tick.value)}
            y={innerHeight + 16}
            textAnchor="middle"
          >
            {tick.label}
          </text>
        ))}

        <line className="chart-axis-line" x1="0" x2="0" y1="0" y2={innerHeight} />
        <line className="chart-axis-line" x1="0" x2={innerWidth} y1={innerHeight} y2={innerHeight} />

        {series.map((s) => (
          <g key={s.name}>
            <path
              d={pathFor(s.points)}
              fill="none"
              stroke={s.color}
              strokeWidth="1.75"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {s.points
              .filter((p) => p.y !== null && p.y !== undefined)
              .map((p, i) => (
                <circle
                  key={i}
                  cx={scaleX(p.x)}
                  cy={scaleY(p.y)}
                  r="2.4"
                  fill={s.color}
                  data-testid="chart-point"
                  data-series={s.name}
                />
              ))}
          </g>
        ))}

        {xLabel && (
          <text className="chart-axis-title" x={innerWidth / 2} y={innerHeight + 32} textAnchor="middle">
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            className="chart-axis-title"
            transform={`translate(${-MARGIN.left + 12}, ${innerHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {yLabel}
          </text>
        )}
      </g>
    </svg>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span key={item.name} className="legend-item">
          <span className="legend-swatch" style={{ background: item.color }} />
          {item.name}
        </span>
      ))}
    </div>
  );
}
