import { Fragment, useEffect, useState } from "react";
import AnalyticsView from "./analytics/AnalyticsView";
import { METHOD_LABELS, fieldsForMethod, methodAfterExerciseChange, methodsForExercise } from "./methodRules";
import { priceOption } from "./api";

const CONTRACT_FIELDS = [
  { key: "S", label: "Stock Price (S):", step: "1", defaultValue: "100" },
  { key: "K", label: "Strike Price (K):", step: "1", defaultValue: "100" },
  { key: "T", label: "Time to Expiry (T in years):", step: "0.1", defaultValue: "1" },
  { key: "r", label: "Risk-Free Rate (r as decimal):", step: "0.01", defaultValue: "0.05" },
  { key: "y", label: "Yield Rate (y as decimal):", step: "0.01", defaultValue: "0" },
  { key: "sigma", label: "Volatility (σ as decimal):", step: "0.01", defaultValue: "0.15" },
];

const SIZING_FIELDS = {
  time_steps: { label: "Time steps:", step: "10", defaultValue: "300" },
  seed: { label: "Random seed:", step: "1", defaultValue: "42" },
  iterations: { label: "Iterations (n):", step: "100", defaultValue: "10000" },
  timestep: { label: "Timestep (dt):", step: "0.001", defaultValue: "0.004" },
};

//: Price leads on its own row; the Greeks follow as a uniform grid, since
//: they are read against each other rather than against the price.
const GREEK_LABELS = {
  delta: "Delta Δ",
  gamma: "Gamma Γ",
  theta: "Theta Θ",
  vega: "Vega 𝓋",
  rho: "Rho ρ",
};

function defaultFieldState() {
  const state = {};
  for (const field of CONTRACT_FIELDS) state[field.key] = field.defaultValue;
  for (const key of Object.keys(SIZING_FIELDS)) state[key] = SIZING_FIELDS[key].defaultValue;
  return state;
}

function formatResult(value) {
  return value === null || value === undefined ? "N/A" : value.toFixed(3);
}

function Stat({ label, value, hero = false }) {
  const text = formatResult(value);
  return (
    <div className={hero ? "stat stat-hero" : "stat"}>
      <div className="stat-label">{label}</div>
      <div className={text === "N/A" ? "stat-value is-na" : "stat-value"}>{text}</div>
    </div>
  );
}

function Segmented({ legend, name, options, value, onChange }) {
  return (
    <div className="field">
      <span className="control-label">{legend}</span>
      <div className="segmented" role="group" aria-label={legend}>
        {options.map((option) => (
          // Fragment, not a wrapper element: the checked styling keys off the
          // `input:checked + label` adjacency, and the labels are flex items
          // of `.segmented` itself.
          <Fragment key={option.id}>
            <input
              type="radio"
              id={option.id}
              name={name}
              autoComplete="off"
              checked={value === option.value}
              onChange={() => onChange(option.value)}
            />
            <label htmlFor={option.id}>{option.label}</label>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

/** The visible tab is kept in the URL hash so a view is linkable and survives
 *  a refresh -- without it, "look at the SPY surface" is not a thing you can
 *  send someone. Falls back to the calculator for an empty or unknown hash. */
function viewFromHash() {
  return typeof window !== "undefined" && window.location.hash === "#analytics"
    ? "analytics"
    : "calculator";
}

export default function App() {
  const [view, setView] = useState(viewFromHash);
  const [exerciseType, setExerciseType] = useState("european");
  const [optionType, setOptionType] = useState("call");
  const [method, setMethod] = useState("BSM");
  const [fields, setFields] = useState(defaultFieldState);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const visibleSizingKeys = fieldsForMethod(method);

  // Back/forward between the two tabs, not just forward navigation.
  useEffect(() => {
    const onHashChange = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function showView(next) {
    setView(next);
    window.location.hash = next === "analytics" ? "#analytics" : "";
  }

  function handleExerciseTypeChange(newExerciseType) {
    setExerciseType(newExerciseType);
    setMethod((currentMethod) => methodAfterExerciseChange(currentMethod, newExerciseType));
  }

  function handleFieldChange(key, value) {
    setFields((current) => ({ ...current, [key]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setPending(true);
    setError(null);
    setResult(null);

    const payload = {
      exercise_type: exerciseType,
      option_type: optionType,
      method,
    };
    for (const field of CONTRACT_FIELDS) payload[field.key] = Number(fields[field.key]);
    for (const key of visibleSizingKeys) payload[key] = Number(fields[key]);

    try {
      const priced = await priceOption(payload);
      setResult(priced);
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <header className="app-header">
        <span className="wordmark">
          <span className="wordmark-mark">∂V</span>
          Derivative Pricing
          <span className="wordmark-sub">volatility analytics</span>
        </span>

        <ul className="tabs" role="tablist">
          <li role="presentation">
            <button
              type="button"
              role="tab"
              aria-selected={view === "calculator"}
              className="tab"
              onClick={() => showView("calculator")}
            >
              Calculator
            </button>
          </li>
          <li role="presentation">
            <button
              type="button"
              role="tab"
              aria-selected={view === "analytics"}
              className="tab"
              onClick={() => showView("analytics")}
            >
              Analytics
            </button>
          </li>
        </ul>

        <span className="header-spacer" />
        <span className="header-note">6 methods · American &amp; European</span>
      </header>

      <main>
        {view === "analytics" && <AnalyticsView />}

        {view === "calculator" && (
          <div className="split">
            <section className="panel">
              <div className="panel-header">
                <h2 className="panel-title">Contract</h2>
              </div>
              <p className="panel-sub">
                Priced by the library directly; the API bounds every sizing input server-side.
              </p>

              <form onSubmit={handleSubmit}>
                <fieldset disabled={pending}>
                  <div className="control-row">
                    <Segmented
                      legend="Exercise Type:"
                      name="exercise-type"
                      value={exerciseType}
                      onChange={handleExerciseTypeChange}
                      options={[
                        { id: "european", value: "european", label: "European" },
                        { id: "american", value: "american", label: "American" },
                      ]}
                    />
                    <Segmented
                      legend="Option Type:"
                      name="option-type"
                      value={optionType}
                      onChange={setOptionType}
                      options={[
                        { id: "call", value: "call", label: "Call" },
                        { id: "put", value: "put", label: "Put" },
                      ]}
                    />
                    <div className="field">
                      <label htmlFor="method">Evaluation method:</label>
                      <select id="method" value={method} onChange={(event) => setMethod(event.target.value)}>
                        {methodsForExercise(exerciseType).map((value) => (
                          <option key={value} value={value}>
                            {METHOD_LABELS[value]}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="section-label" style={{ marginTop: 22 }}>
                    Parameters
                  </div>
                  <div className="field-grid">
                    {CONTRACT_FIELDS.map((field) => (
                      <div key={field.key} className="field">
                        <label htmlFor={field.key}>{field.label}</label>
                        <input
                          type="number"
                          id={field.key}
                          step={field.step}
                          value={fields[field.key]}
                          onChange={(event) => handleFieldChange(field.key, event.target.value)}
                          required
                        />
                      </div>
                    ))}
                  </div>

                  {visibleSizingKeys.length > 0 && (
                    <>
                      <div className="section-label" style={{ marginTop: 22 }}>
                        {METHOD_LABELS[method]} sizing
                      </div>
                      <div className="field-grid">
                        {visibleSizingKeys.map((key) => (
                          <div key={key} className="field">
                            <label htmlFor={key}>{SIZING_FIELDS[key].label}</label>
                            <input
                              type="number"
                              id={key}
                              step={SIZING_FIELDS[key].step}
                              value={fields[key]}
                              onChange={(event) => handleFieldChange(key, event.target.value)}
                              required
                            />
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  <div style={{ marginTop: 24 }}>
                    <button type="submit" className="btn" disabled={pending}>
                      <span>{pending ? "Calculating..." : "Calculate Price"}</span>
                      {pending && <span className="spinner" data-testid="pending-indicator" aria-hidden="true" />}
                    </button>
                  </div>
                </fieldset>
              </form>
            </section>

            <section className="panel">
              <div className="panel-header">
                <h2 className="panel-title">Result</h2>
                <span className="panel-spacer" />
                <span className="chip">{METHOD_LABELS[method]}</span>
              </div>
              <p className="panel-sub">
                {exerciseType === "american" ? "American" : "European"}{" "}
                {optionType} · Greeks by the method's own estimation
              </p>

              {error && (
                <div className="notice notice-error" role="alert">
                  {error}
                </div>
              )}

              {!error && !result && (
                <div className="placeholder-panel">
                  {pending ? "Pricing…" : "Submit the contract to price it."}
                </div>
              )}

              {result && (
                <div className="stat-grid">
                  <Stat label="Option Value" value={result.price} hero />
                  {Object.keys(GREEK_LABELS).map((key) => (
                    <Stat key={key} label={GREEK_LABELS[key]} value={result[key]} />
                  ))}
                </div>
              )}
            </section>
          </div>
        )}
      </main>

      <footer className="app-footer">© 2024 Jay Singh Chauhan</footer>
    </>
  );
}
