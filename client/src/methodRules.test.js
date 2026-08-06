import { describe, expect, it } from "vitest";
import { fieldsForMethod, methodAfterExerciseChange, methodsForExercise } from "./methodRules";

describe("methodsForExercise", () => {
  it("excludes BSM and MC from American, and includes LSMC only there", () => {
    const american = methodsForExercise("american");
    expect(american).toContain("LSMC");
    expect(american).not.toContain("BSM");
    expect(american).not.toContain("MC");
  });

  it("excludes LSMC from European", () => {
    expect(methodsForExercise("european")).not.toContain("LSMC");
    expect(methodsForExercise("european")).toEqual(expect.arrayContaining(["BSM", "MC"]));
  });
});

describe("methodAfterExerciseChange", () => {
  it("auto-switches BSM to BT when moving to American", () => {
    expect(methodAfterExerciseChange("BSM", "american")).toBe("BT");
  });

  it("auto-switches MC to BT when moving to American", () => {
    expect(methodAfterExerciseChange("MC", "american")).toBe("BT");
  });

  it("auto-switches LSMC to BSM when moving to European", () => {
    expect(methodAfterExerciseChange("LSMC", "european")).toBe("BSM");
  });

  it("leaves a method valid in both exercise types unchanged", () => {
    expect(methodAfterExerciseChange("TT", "american")).toBe("TT");
    expect(methodAfterExerciseChange("TT", "european")).toBe("TT");
  });
});

describe("fieldsForMethod", () => {
  it("shows only time_steps for BT and TT", () => {
    expect(fieldsForMethod("BT")).toEqual(["time_steps"]);
    expect(fieldsForMethod("TT")).toEqual(["time_steps"]);
  });

  it("shows seed, iterations and timestep for MC", () => {
    expect(fieldsForMethod("MC")).toEqual(["seed", "iterations", "timestep"]);
  });

  it("shows seed and iterations but not timestep for LSMC", () => {
    expect(fieldsForMethod("LSMC")).toEqual(["seed", "iterations"]);
  });

  it("shows only timestep for FD", () => {
    expect(fieldsForMethod("FD")).toEqual(["timestep"]);
  });

  it("shows no sizing fields for BSM", () => {
    expect(fieldsForMethod("BSM")).toEqual([]);
  });
});
