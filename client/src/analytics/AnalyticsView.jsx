import { useEffect, useState } from "react";
import LineChart from "./LineChart";
import { fetchComparison, fetchSurface, fetchSymbols, fetchViolations } from "./api";

const IMPLIED_COLOR = "#0d6efd";
const REALIZED_COLOR = "#dc3545";

function formatVol(value) {
  return value === null || value === undefined ? "—" : value.toFixed(4);
}

function CaptureTimestamp({ value }) {
  return (
    <p className="text-muted">
      {value === null
        ? "Not yet captured for this symbol."
        : `Snapshot captured ${new Date(value).toLocaleString()}`}
    </p>
  );
}

function InsufficientData({ caption }) {
  return (
    <div className="border rounded p-3 text-muted small" role="note">
      Insufficient data — {caption}
    </div>
  );
}

function SkewSection({ curves }) {
  if (curves.length === 0) return <p className="text-muted">No skew curves for this symbol yet.</p>;
  return (
    <div className="row">
      {curves.map((curve) => (
        <div key={curve.expiry} className="col-md-6 mb-4">
          <h5>Skew — {curve.expiry}</h5>
          {curve.status === "ok" ? (
            <>
              <LineChart
                ariaLabel={`Skew curve for expiry ${curve.expiry}`}
                series={[
                  {
                    name: "implied_vol",
                    color: IMPLIED_COLOR,
                    points: curve.points.map((p) => ({ x: p.strike, y: p.implied_vol })),
                  },
                ]}
              />
              <p className="small text-muted mb-0">
                {curve.points.map((p) => `${p.strike} (${formatVol(p.implied_vol)})`).join(", ")}
              </p>
            </>
          ) : (
            <InsufficientData caption={`too few valid quotes on the ${curve.expiry} expiry to plot a skew curve.`} />
          )}
        </div>
      ))}
    </div>
  );
}

function TermStructureSection({ curves }) {
  if (curves.length === 0) {
    return <p className="text-muted">No term-structure curves for this symbol yet.</p>;
  }
  return (
    <div className="row">
      {curves.map((curve) => (
        <div key={curve.strike} className="col-md-6 mb-4">
          <h5>Term structure — strike {curve.strike}</h5>
          {curve.status === "ok" ? (
            <LineChart
              ariaLabel={`Term structure curve for strike ${curve.strike}`}
              series={[
                {
                  name: "implied_vol",
                  color: IMPLIED_COLOR,
                  points: curve.points.map((p, i) => ({ x: i, y: p.implied_vol })),
                },
              ]}
            />
          ) : (
            <InsufficientData
              caption={`too few live expiries at strike ${curve.strike} to plot a term-structure line.`}
            />
          )}
          {curve.status === "ok" && (
            <p className="small text-muted mb-0">
              {curve.points.map((p) => `${p.expiry} (${formatVol(p.implied_vol)})`).join(", ")}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

function ComparisonSection({ points }) {
  if (points.length === 0) {
    return <p className="text-muted">Not enough accumulated history yet to compare implied and realized volatility.</p>;
  }
  return (
    <div>
      <LineChart
        ariaLabel="Implied versus realized volatility over time"
        series={[
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
        ]}
      />
      <p className="small text-muted mb-0">
        <span style={{ color: IMPLIED_COLOR }}>●</span> implied &nbsp;
        <span style={{ color: REALIZED_COLOR }}>●</span> realized &nbsp;
        {points[0].as_of} – {points[points.length - 1].as_of}
      </p>
      <p className="small text-muted mb-0">
        Latest ({points[points.length - 1].as_of}): implied {formatVol(points[points.length - 1].implied_vol)},
        realized {formatVol(points[points.length - 1].realized_vol)}
      </p>
    </div>
  );
}

function ViolationsSection({ count, violations }) {
  return (
    <div>
      <h5>Violations ({count})</h5>
      {count === 0 ? (
        <p className="text-muted">No no-arbitrage violations recorded for this snapshot.</p>
      ) : (
        <div className="table-responsive">
          <table className="table table-sm table-hover" aria-label="Recorded no-arbitrage violations">
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
                  <td>{row.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

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
      <div className="alert alert-danger" role="alert">
        {symbolsError}
      </div>
    );
  }

  if (!symbol) {
    return <p role="status">Loading symbols…</p>;
  }

  return (
    <div>
      <div className="row align-items-center mb-3">
        <div className="col-md-4">
          <label className="form-label text-muted" style={{ fontSize: "0.85rem" }} htmlFor="analytics-symbol">
            Symbol:
          </label>
          <select
            id="analytics-symbol"
            className="form-select"
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
      </div>

      {status === "loading" && <p role="status">Loading analytics for {symbol}…</p>}

      {status === "error" && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      {status === "ready" && surface && comparison && violations && (
        <div>
          <CaptureTimestamp value={surface.capture_time} />

          {surface.capture_time === null ? (
            <p className="text-muted">No data captured yet for {symbol}.</p>
          ) : (
            <>
              <h4>Skew</h4>
              <SkewSection curves={surface.skew} />

              <h4>Term structure</h4>
              <TermStructureSection curves={surface.term_structure} />
            </>
          )}

          <h4>Implied vs. realized volatility</h4>
          <ComparisonSection points={comparison.points} />

          <ViolationsSection count={violations.count} violations={violations.violations} />
        </div>
      )}
    </div>
  );
}
