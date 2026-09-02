// Which methods and sizing fields apply to a given exercise type and method —
// pulled out of the component as pure functions so the two behaviors U15
// carries over from templates/index.html (per-method field toggling,
// exercise-type filtering with auto-switch) are each one obvious thing to
// test, not an assertion buried in a render tree.

export const METHOD_LABELS = {
  BSM: "Black-Scholes-Merton model",
  BT: "Binomial Tree method",
  TT: "Trinomial Tree method",
  MC: "Monte Carlo simulation",
  LSMC: "Least Squares Monte Carlo simulation",
  FD: "Finite Difference method",
};

const EUROPEAN_METHODS = ["BSM", "BT", "TT", "MC", "FD"];
const AMERICAN_METHODS = ["BT", "TT", "LSMC", "FD"];

export function methodsForExercise(exerciseType) {
  return exerciseType === "american" ? AMERICAN_METHODS : EUROPEAN_METHODS;
}

// The default each exercise type falls back to when the current method is
// not in its list — same defaults templates/index.html used (BT for
// American, BSM for European).
const FALLBACK_METHOD = { european: "BSM", american: "BT" };

export function methodAfterExerciseChange(currentMethod, newExerciseType) {
  const allowed = methodsForExercise(newExerciseType);
  return allowed.includes(currentMethod) ? currentMethod : FALLBACK_METHOD[newExerciseType];
}

// Sizing fields each method reads, in the shape the API expects them under
// (see api/pricing_routes.py). BSM and the base contract fields (S, K, T, r,
// sigma, y) need nothing extra — an empty list is a real answer, not a gap.
export function fieldsForMethod(method) {
  switch (method) {
    case "BT":
    case "TT":
      return ["time_steps"];
    case "MC":
      return ["seed", "iterations", "timestep"];
    case "LSMC":
      return ["seed", "iterations"];
    case "FD":
      return ["timestep"];
    default:
      return [];
  }
}
