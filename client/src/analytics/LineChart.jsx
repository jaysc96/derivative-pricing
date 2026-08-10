// Minimal hand-rolled SVG line chart -- R19 needs two 2D views, not a 3D
// mesh, and the surface, term structure, and implied-vs-realized views are
// otherwise identical shapes (one or two series of x/y points against a
// linear scale). A charting dependency would buy little over this for three
// call sites, none of which needs zoom, tooltips, or more than a couple of
// series.
//
// `series` is `[{ name, color, points: [{ x, y }] }]`, `y` may be `null` --
// the comparison view's two series do not always both have a value on the
// same day, and a null breaks the line into segments rather than drawing a
// point at 0.
export default function LineChart({ series, width = 480, height = 200, ariaLabel }) {
  const margin = { top: 10, right: 12, bottom: 10, left: 12 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const valid = series.flatMap((s) => s.points.filter((p) => p.y !== null && p.y !== undefined));
  if (valid.length === 0) return null;

  const xs = valid.map((p) => p.x);
  const ys = valid.map((p) => p.y);
  const xDomain = [Math.min(...xs), Math.max(...xs)];
  const yDomain = [Math.min(...ys), Math.max(...ys)];
  const yPad = (yDomain[1] - yDomain[0]) * 0.1 || Math.abs(yDomain[0]) * 0.1 || 0.01;

  function scaleX(x) {
    const span = xDomain[1] - xDomain[0] || 1;
    return ((x - xDomain[0]) / span) * innerWidth;
  }

  function scaleY(y) {
    const lo = yDomain[0] - yPad;
    const hi = yDomain[1] + yPad;
    const span = hi - lo || 1;
    return innerHeight - ((y - lo) / span) * innerHeight;
  }

  function pathFor(points) {
    const segments = [[]];
    for (const p of points) {
      if (p.y === null || p.y === undefined) {
        if (segments[segments.length - 1].length) segments.push([]);
      } else {
        segments[segments.length - 1].push(p);
      }
    }
    return segments
      .filter((segment) => segment.length > 0)
      .map((segment) =>
        segment.map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" "),
      )
      .join(" ");
  }

  return (
    <svg role="img" aria-label={ariaLabel} width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <g transform={`translate(${margin.left}, ${margin.top})`}>
        {series.map((s) => (
          <g key={s.name}>
            <path d={pathFor(s.points)} fill="none" stroke={s.color} strokeWidth="2" />
            {s.points
              .filter((p) => p.y !== null && p.y !== undefined)
              .map((p, i) => (
                <circle
                  key={i}
                  cx={scaleX(p.x)}
                  cy={scaleY(p.y)}
                  r="3"
                  fill={s.color}
                  data-testid={`chart-point-${s.name}`}
                />
              ))}
          </g>
        ))}
      </g>
    </svg>
  );
}
