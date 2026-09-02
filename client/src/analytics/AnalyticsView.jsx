import { Component, Suspense, lazy, useEffect, useState } from "react";
import LineChart, { Legend } from "./LineChart";
import { fetchComparison, fetchSurface, fetchSymbols, fetchViolations } from "./api";

// Plotly's gl3d bundle alone is ~600kB gzipped -- code-split so it's fetched
// only once someone actually opens the Analytics tab, not on every load of
// the calculator (the default view, and the one most visits never leave).
const Surface3D = lazy(() => import("./Surface3D"));

const IMPLIED_COLOR = "#58a6ff";
const REALIZED_COLOR = "#f0883e";

//: One color per expiry, shared by the 2D skew chart and the 3D surface. A
//: real capture carries a handful of live expiries, so this cycles rather
//: than scaling forever.
const EXPIRY_COLORS = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#39c5cf", "#f778ba"];

/** One color per expiry, keyed by expiry so both views agree on which color
 *  belongs to which expiry regardless of how each one filters curves for
 *  its own display. The basis is every curve with at least one point --
 *  the 3D surface plots those even when the 2D chart's stricter 3-point
 *  floor marks the same curve insufficient-data. Building the assignment
 *  from a narrower, status-filtered list here (as the 2D chart used to)
 *  shifts every later expiry's color out of sync between the two views
 *  the moment a thin expiry sits before a full one. */
function expiryColorMap(curves) {
  const withPoints = curves.filter((c) => c.points.length > 0);
  const map = new Map(withPoints.map((c, i) => [c.expiry, EXPIRY_COLORS[i % EXPIRY_COLORS.length]]));
  return (expiry) => map.get(expiry) ?? EXPIRY_COLORS[0];
}

const asPercent = (v) => `${(v * 100).toFixed(0)}%`;

function formatVol(value) {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

/** ~`count` evenly spaced ticks from a categorical (index-keyed) series. */
function sampledTicks(labels, count = 5) {
  if (labels.length === 0) return [];
  const stride = Math.max(1, Math.ceil(labels.length / count));
  const ticks = [];
  for (let i = 0; i < labels.length; i += stride) {
    ticks.push({ value: i, label: labels[i] });
  }
  return ticks;
}

function InsufficientData({ children }) {
  return (
    <div className="notice notice-inline" data-testid="insufficient-data">
      <span>
        <span className="notice-title">Insufficient data</span> — {children}
      </span>
    </div>
  );
}

// ------------------------------------------------------------- 3D surface

/** This diff introduces the app's only React.lazy/Suspense usage, and
 *  nothing upstream catches a failed chunk load or an in-render exception --
 *  without this, either would unmount the whole page (calculator included),
 *  not just the one panel that failed. Must be a class component; React has
 *  no hook-based equivalent to getDerivedStateFromError. */
class ChartErrorBoundary extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="notice notice-error" role="alert">
          The 3D renderer failed to load. Try reloading the page.
        </div>
      );
    }
    return this.props.children;
  }
}

function Surface3DPanel({ curves, colorFor }) {
  const withData = curves.filter((c) => c.points.length > 0);
  return (
    <section className="panel" aria-label="3D surface">
      <div className="panel-header">
        <h2 className="panel-title">3D surface</h2>
        <span className="panel-spacer" />
        <span className="chip">{withData.length} expiries plotted</span>
      </div>
      <p className="panel-sub">
        Drag to rotate. The mesh only fills between quotes that were actually close together —
        a strike range nothing was observed at renders as a gap, not a smoothed-over guess.
      </p>
      <ChartErrorBoundary>
        <Suspense fallback={<div className="placeholder-panel">Loading the 3D renderer…</div>}>
          <Surface3D curves={curves} colorFor={colorFor} />
        </Suspense>
      </ChartErrorBoundary>
    </section>
  );
}

// ---------------------------------------------------------------- skew

function SkewPanel({ curves, colorFor }) {
  const ok = curves.filter((c) => c.status === "ok");
  const short = curves.filter((c) => c.status !== "ok");

  const series = ok.map((curve) => ({
    name: curve.expiry,
    color: colorFor(curve.expiry),
    points: curve.points.map((p) => ({ x: p.strike, y: p.implied_vol })),
  }));

  return (
    <section className="panel" aria-label="Skew">
      <div className="panel-header">
        <h2 className="panel-title">Skew</h2>
        <span className="panel-spacer" />
        <span className="chip">{ok.length} live expiries</span>
      </div>
      <p className="panel-sub">Implied volatility across strikes, one line per expiry.</p>

      {series.length > 0 ? (
        <>
          <LineChart
            ariaLabel="Implied volatility skew across strikes"
            series={series}
            height={260}
            xLabel="Strike"
            yLabel="Implied vol"
            yFormat={asPercent}
            xFormat={(v) => v.toFixed(0)}
            // Deep out-of-the-money strikes are thinly quoted -- a gap of
            // several times the typical one-strike spacing means nothing was
            // observed in between, so the line breaks there instead of
            // interpolating a shape the archive never actually saw.
            maxGapMultiple={4}
          />
          <Legend items={series} />
        </>
      ) : (
        <div className="placeholder-panel">No skew curves for this symbol yet.</div>
      )}

      {short.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <InsufficientData>
            too few valid quotes to plot {short.length === 1 ? "the" : "these"} {short.length}{" "}
            {short.length === 1 ? "expiry" : "expiries"}: {short.map((c) => c.expiry).join(", ")}. The
            remaining curves above are unaffected.
          </InsufficientData>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------ term structure

function TermStructurePanel({ curves }) {
  const [choice, setChoice] = useState(null);
  if (curves.length === 0) {
    return (
      <section className="panel" aria-label="Term structure">
        <div className="panel-header">
          <h2 className="panel-title">Term structure</h2>
        </div>
        <div className="placeholder-panel">No term-structure curves for this symbol yet.</div>
      </section>
    );
  }

  const strikes = curves.map((c) => String(c.strike));
  // Derived rather than reset in an effect: when the symbol changes the
  // strike list changes under us, and falling back is both correct and one
  // less state transition to get wrong. The fallback is the first strike
  // that actually has a curve, not the numerically lowest -- the low wing
  // is exactly where quotes are thinnest, so defaulting to `strikes[0]`
  // opened the panel on an insufficient-data notice almost every time.
  const firstPlottable = curves.find((c) => c.status === "ok") ?? curves[0];
  const active = strikes.includes(choice) ? choice : String(firstPlottable.strike);
  const curve = curves.find((c) => String(c.strike) === active);

  return (
    <section className="panel" aria-label="Term structure">
      <div className="panel-header">
        <h2 className="panel-title">Term structure</h2>
        <span className="panel-spacer" />
        <span className="chip">{curves.length} strikes</span>
      </div>
      <p className="panel-sub">Implied volatility across expiries, one strike at a time.</p>

      <div className="toolbar">
        <div className="field">
          <label htmlFor="term-strike">Strike:</label>
          <select id="term-strike" value={active} onChange={(event) => setChoice(event.target.value)}>
            {curves.map((c) => (
              <option key={c.strike} value={String(c.strike)}>
                {c.strike}
                {c.status === "ok" ? "" : " — insufficient"}
              </option>
            ))}
          </select>
        </div>
      </div>

      {curve.status === "ok" ? (
        <>
          <LineChart
            ariaLabel={`Term structure at strike ${curve.strike}`}
            series={[
              {
                name: `strike ${curve.strike}`,
                color: IMPLIED_COLOR,
                points: curve.points.map((p, i) => ({ x: i, y: p.implied_vol })),
              },
            ]}
            height={240}
            xTicks={sampledTicks(curve.points.map((p) => p.expiry), 4)}
            xLabel="Expiry"
            yLabel="Implied vol"
            yFormat={asPercent}
          />
        </>
      ) : (
        <InsufficientData>
          too few live expiries at strike {curve.strike} to plot a term-structure line. Pick another
          strike above.
        </InsufficientData>
      )}
    </section>
  );
}

// ------------------------------------------------------------ comparison

function ComparisonPanel({ points }) {
  if (points.length === 0) {
    return (
      <section className="panel" aria-label="Implied versus realized">
        <div className="panel-header">
          <h2 className="panel-title">Implied vs. realized</h2>
        </div>
        <div className="placeholder-panel">
          Not enough accumulated history yet to compare implied and realized volatility.
        </div>
      </section>
    );
  }

  const latest = points[points.length - 1];
  const impliedCount = points.filter((p) => p.implied_vol !== null).length;
  const series = [
    {
      name: "implied",
      color: IMPLIED_COLOR,
      points: points.map((p, i) => ({ x: i, y: p.implied_vol })),
    },
    {
      name: "realized",
      color: REALIZED_COLOR,
      points: points.map((p, i) => ({ x: i, y: p.realized_vol })),
    },
  ];

  return (
    <section className="panel" aria-label="Implied versus realized">
      <div className="panel-header">
        <h2 className="panel-title">Implied vs. realized</h2>
        <span className="panel-spacer" />
        <span className="chip">{points.length} days</span>
      </div>
      <p className="panel-sub">
        At-the-money call implied volatility on the nearest usable expiry, against 21-day
        realized volatility from underlying history.
      </p>

      {impliedCount < points.length && (
        <div className="notice notice-inline" style={{ marginBottom: 14 }}>
          Implied volatility is available for {impliedCount} of {points.length} days —{" "}
          {impliedCount === 1 ? "it" : "the cluster on the right"} will fill in as more
          scheduled captures accumulate. Realized volatility has a longer history because it
          only needs underlying price bars, not an option chain.
        </div>
      )}

      <LineChart
        ariaLabel="Implied versus realized volatility over time"
        series={series}
        height={240}
        xTicks={sampledTicks(points.map((p) => p.as_of), 5)}
        yLabel="Volatility"
        yFormat={asPercent}
      />
      <Legend items={series} />
      <div className="stat-grid" style={{ marginTop: 16 }}>
        <div className="stat">
          <div className="stat-label">Implied ({latest.as_of})</div>
          <div className="stat-value">{formatVol(latest.implied_vol)}</div>
        </div>
        <div className="stat">
          <div className="stat-label">Realized ({latest.as_of})</div>
          <div className="stat-value">{formatVol(latest.realized_vol)}</div>
        </div>
      </div>
    </section>
  );
}

// ------------------------------------------------------------ violations

function ViolationsPanel({ count, violations }) {
  return (
    <section className="panel" aria-label="Violations">
      <div className="panel-header">
        <h2 className="panel-title">No-arbitrage violations</h2>
        <span className="panel-spacer" />
        <span className={count === 0 ? "chip chip-ok" : "chip chip-danger"}>
          <span className="chip-dot" />
          {count} rejected {count === 1 ? "leg" : "legs"}
        </span>
      </div>
      <p className="panel-sub">
        Legs excluded from the surface above by the quality gate, shown rather than dropped.
      </p>

      {count === 0 ? (
        <div className="placeholder-panel">No no-arbitrage violations recorded for this snapshot.</div>
      ) : (
        <div className="table-scroll">
          <table aria-label="Recorded no-arbitrage violations">
            <thead>
              <tr>
                <th>Expiry</th>
                <th>Strike</th>
                <th>Side</th>
                <th>Bound</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {violations.map((row) => (
                <tr key={row.contract_symbol}>
                  <td>{row.expiry}</td>
                  <td>{row.strike}</td>
                  <td>{row.option_type}</td>
                  <td>{row.kind}</td>
                  <td className="cell-detail">{row.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ view

export default function AnalyticsView() {
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState(null);
  const [symbolsError, setSymbolsError] = useState(null);

  const [status, setStatus] = useState("loading");
  const [surface, setSurface] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [violations, setViolations] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetchSymbols()
      .then((body) => {
        if (cancelled) return;
        setSymbols(body.symbols);
        setSymbol(body.default);
      })
      .catch((err) => {
        if (!cancelled) setSymbolsError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!symbol) return undefined;
    let cancelled = false;

    // One combined loading state across all three requests, and the
    // previous symbol's charts cleared immediately -- a partially-landed
    // render (say, surface back but comparison still in flight) must never
    // read as a complete picture of the newly selected symbol.
    setStatus("loading");
    setSurface(null);
    setComparison(null);
    setViolations(null);
    setError(null);

    Promise.all([fetchSurface(symbol), fetchComparison(symbol), fetchViolations(symbol)])
      .then(([surfaceBody, comparisonBody, violationsBody]) => {
        if (cancelled) return;
        setSurface(surfaceBody);
        setComparison(comparisonBody);
        setViolations(violationsBody);
        setStatus("ready");
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err.message);
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (symbolsError) {
    return (
      <div className="notice notice-error" role="alert">
        {symbolsError}
      </div>
    );
  }

  if (!symbol) {
    return (
      <div className="loading-row" role="status">
        <span className="spinner" /> Loading symbols…
      </div>
    );
  }

  // Computed once per render and handed to both charts, so a symbol with an
  // odd number of expiries can't produce two different assignments for the
  // same expiry between them.
  const colorFor = surface ? expiryColorMap(surface.skew) : null;

  return (
    <div>
      <div className="toolbar">
        <div className="field">
          <label htmlFor="analytics-symbol">Symbol:</label>
          <select
            id="analytics-symbol"
            aria-label="Symbol"
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
          >
            {symbols.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>

        {status === "ready" && surface && (
          <span className="chip">
            {surface.capture_time === null
              ? "Not yet captured for this symbol."
              : `Snapshot captured ${new Date(surface.capture_time).toLocaleString()}`}
          </span>
        )}
      </div>

      {status === "loading" && (
        <div className="loading-row" role="status">
          <span className="spinner" /> Loading analytics for {symbol}…
        </div>
      )}

      {status === "error" && (
        <div className="notice notice-error" role="alert">
          {error}
        </div>
      )}

      {status === "ready" && surface && comparison && violations && (
        <>
          {surface.capture_time === null ? (
            <div className="panel">
              <div className="placeholder-panel">No data captured yet for {symbol}.</div>
            </div>
          ) : (
            <>
              <Surface3DPanel curves={surface.skew} colorFor={colorFor} />
              <SkewPanel curves={surface.skew} colorFor={colorFor} />
              <TermStructurePanel curves={surface.term_structure} />
            </>
          )}
          <ComparisonPanel points={comparison.points} />
          <ViolationsPanel count={violations.count} violations={violations.violations} />
        </>
      )}
    </div>
  );
}
