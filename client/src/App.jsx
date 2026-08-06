import { useState } from "react";
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

const GREEK_LABELS = {
  price: "Option Value",
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

export default function App() {
  const [exerciseType, setExerciseType] = useState("european");
  const [optionType, setOptionType] = useState("call");
  const [method, setMethod] = useState("BSM");
  const [fields, setFields] = useState(defaultFieldState);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const visibleSizingKeys = fieldsForMethod(method);

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
    <div className="container">
      <header className="text-center p-3">
        <h1 className="mb-5">Option Pricing Calculator</h1>
      </header>

      <form onSubmit={handleSubmit}>
        <div className="row">
          <div className="col-md-4 mb-3">
            <label className="form-label text-muted" style={{ fontSize: "0.85rem" }}>
              Exercise Type:
            </label>
            <div className="btn-group d-flex" role="group" aria-label="Exercise Type">
              <input
                type="radio"
                className="btn-check"
                id="european"
                autoComplete="off"
                checked={exerciseType === "european"}
                onChange={() => handleExerciseTypeChange("european")}
              />
              <label className="btn btn-outline-secondary" htmlFor="european">
                European
              </label>
              <input
                type="radio"
                className="btn-check"
                id="american"
                autoComplete="off"
                checked={exerciseType === "american"}
                onChange={() => handleExerciseTypeChange("american")}
              />
              <label className="btn btn-outline-info" htmlFor="american">
                American
              </label>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <label className="form-label text-muted" style={{ fontSize: "0.85rem" }}>
              Option Type:
            </label>
            <div className="btn-group d-flex" role="group" aria-label="Option Type">
              <input
                type="radio"
                className="btn-check"
                id="call"
                autoComplete="off"
                checked={optionType === "call"}
                onChange={() => setOptionType("call")}
              />
              <label className="btn btn-outline-success" htmlFor="call">
                Call
              </label>
              <input
                type="radio"
                className="btn-check"
                id="put"
                autoComplete="off"
                checked={optionType === "put"}
                onChange={() => setOptionType("put")}
              />
              <label className="btn btn-outline-danger" htmlFor="put">
                Put
              </label>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="form-floating">
              <select
                id="method"
                className="form-select"
                value={method}
                onChange={(event) => setMethod(event.target.value)}
              >
                {methodsForExercise(exerciseType).map((value) => (
                  <option key={value} value={value}>
                    {METHOD_LABELS[value]}
                  </option>
                ))}
              </select>
              <label htmlFor="method" className="form-label">
                Evaluation method:
              </label>
            </div>
          </div>
        </div>

        <div className="row">
          {CONTRACT_FIELDS.map((field) => (
            <div key={field.key} className="col-md-4 mb-3">
              <div className="form-floating">
                <input
                  type="number"
                  id={field.key}
                  step={field.step}
                  className="form-control"
                  value={fields[field.key]}
                  onChange={(event) => handleFieldChange(field.key, event.target.value)}
                  required
                />
                <label htmlFor={field.key} className="form-label">
                  {field.label}
                </label>
              </div>
            </div>
          ))}
        </div>

        {visibleSizingKeys.length > 0 && (
          <div className="row">
            {visibleSizingKeys.map((key) => (
              <div key={key} className="col-md-4 mb-3">
                <div className="form-floating">
                  <input
                    type="number"
                    id={key}
                    step={SIZING_FIELDS[key].step}
                    className="form-control"
                    value={fields[key]}
                    onChange={(event) => handleFieldChange(key, event.target.value)}
                    required
                  />
                  <label htmlFor={key} className="form-label">
                    {SIZING_FIELDS[key].label}
                  </label>
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="row align-items-center">
          <div className="col-md-6 mb-3">
            <button type="submit" className="btn btn-primary" disabled={pending}>
              <span>{pending ? "Calculating..." : "Calculate Price"}</span>
              {pending && (
                <span
                  className="spinner-grow spinner-grow-sm ms-2"
                  role="status"
                  aria-hidden="true"
                  data-testid="pending-indicator"
                />
              )}
            </button>
          </div>
        </div>
      </form>

      {error && (
        <div className="alert alert-danger" role="alert">
          {error}
        </div>
      )}

      {result && (
        <div className="container">
          <h3 className="mt-3">Results:</h3>
          <div className="table-responsive">
            <table className="table table-hover table-striped table-bordered">
              <thead>
                <tr>
                  {Object.keys(GREEK_LABELS).map((key) => (
                    <th key={key}>{GREEK_LABELS[key]}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  {Object.keys(GREEK_LABELS).map((key) => (
                    <td key={key}>{result[key].toFixed(3)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      <footer className="text-center p-3 mt-5">
        <p>&copy; 2024 Jay Singh Chauhan</p>
      </footer>
    </div>
  );
}
