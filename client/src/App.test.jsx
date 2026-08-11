import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { act } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { priceOption } from "./api";

vi.mock("./api", () => ({ priceOption: vi.fn() }));

// App imports AnalyticsView unconditionally (the calculator/analytics split
// is a runtime tab, not a code split). Most tests here never switch off the
// calculator tab, so AnalyticsView's own data-fetching and its Surface3D
// child (real Plotly, no WebGL in jsdom -- see analytics/AnalyticsView.test.jsx)
// are stubbed out; App's responsibility is which top-level view is shown,
// not what AnalyticsView does once mounted -- that's AnalyticsView.test.jsx's job.
vi.mock("./analytics/AnalyticsView", () => ({
  default: () => <div data-testid="analytics-view-stub" />,
}));

beforeEach(() => {
  priceOption.mockReset();
  window.location.hash = "";
});

function methodSelect() {
  return screen.getByLabelText("Evaluation method:");
}

// --------------------------------------------------------------------------
// Per-method field toggling
// --------------------------------------------------------------------------

describe("per-method field toggling", () => {
  it("shows only the fields Monte Carlo uses", async () => {
    render(<App />);
    await userEvent.selectOptions(methodSelect(), "MC");

    expect(screen.getByLabelText("Random seed:")).toBeInTheDocument();
    expect(screen.getByLabelText("Iterations (n):")).toBeInTheDocument();
    expect(screen.getByLabelText("Timestep (dt):")).toBeInTheDocument();
    expect(screen.queryByLabelText("Time steps:")).not.toBeInTheDocument();
  });

  it("shows only time steps for the Binomial Tree", async () => {
    render(<App />);
    await userEvent.selectOptions(methodSelect(), "BT");

    expect(screen.getByLabelText("Time steps:")).toBeInTheDocument();
    expect(screen.queryByLabelText("Random seed:")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Timestep (dt):")).not.toBeInTheDocument();
  });

  it("shows no sizing fields at all for Black-Scholes-Merton", () => {
    render(<App />);

    expect(screen.queryByLabelText("Random seed:")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Time steps:")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Timestep (dt):")).not.toBeInTheDocument();
  });

  it("shows seed and iterations but not timestep for LSMC", async () => {
    render(<App />);
    await userEvent.click(screen.getByLabelText("American"));
    await userEvent.selectOptions(methodSelect(), "LSMC");

    expect(screen.getByLabelText("Random seed:")).toBeInTheDocument();
    expect(screen.getByLabelText("Iterations (n):")).toBeInTheDocument();
    expect(screen.queryByLabelText("Timestep (dt):")).not.toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------
// Exercise-type filtering and auto-switch
// --------------------------------------------------------------------------

describe("exercise-type filtering", () => {
  it("removes BSM and MC from the method list under American exercise", async () => {
    render(<App />);
    await userEvent.click(screen.getByLabelText("American"));

    const options = within(methodSelect()).getAllByRole("option").map((o) => o.value);
    expect(options).not.toContain("BSM");
    expect(options).not.toContain("MC");
    expect(options).toContain("LSMC");
  });

  it("auto-switches away from BSM when American is selected", async () => {
    render(<App />);
    expect(methodSelect()).toHaveValue("BSM");

    await userEvent.click(screen.getByLabelText("American"));

    expect(methodSelect()).toHaveValue("BT");
  });

  it("auto-switches away from LSMC when European is selected again", async () => {
    render(<App />);
    await userEvent.click(screen.getByLabelText("American"));
    await userEvent.selectOptions(methodSelect(), "LSMC");

    await userEvent.click(screen.getByLabelText("European"));

    expect(methodSelect()).toHaveValue("BSM");
  });

  it("leaves a method valid under both exercise types alone when switching", async () => {
    render(<App />);
    await userEvent.selectOptions(methodSelect(), "TT");

    await userEvent.click(screen.getByLabelText("American"));

    expect(methodSelect()).toHaveValue("TT");
  });
});

// --------------------------------------------------------------------------
// Pending state
// --------------------------------------------------------------------------

describe("the pending state", () => {
  it("disables submit and shows an indicator from request start until the response resolves", async () => {
    let resolvePrice;
    priceOption.mockReturnValue(
      new Promise((resolve) => {
        resolvePrice = resolve;
      }),
    );
    render(<App />);
    const submit = screen.getByRole("button", { name: /calculate price/i });

    await userEvent.click(submit);

    expect(submit).toBeDisabled();
    expect(screen.getByTestId("pending-indicator")).toBeInTheDocument();

    resolvePrice({ price: 10, delta: 0.5, gamma: 0.1, theta: -0.05, vega: 0.2, rho: 0.1 });

    await waitFor(() => expect(submit).not.toBeDisabled());
    expect(screen.queryByTestId("pending-indicator")).not.toBeInTheDocument();
  });

  it("locks every field, not just submit, for the whole in-flight window", async () => {
    let resolvePrice;
    priceOption.mockReturnValue(
      new Promise((resolve) => {
        resolvePrice = resolve;
      }),
    );
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /calculate price/i }));

    expect(methodSelect()).toBeDisabled();
    expect(screen.getByLabelText("Stock Price (S):")).toBeDisabled();
    expect(screen.getByLabelText("American")).toBeDisabled();

    resolvePrice({ price: 10, delta: 0.5, gamma: 0.1, theta: -0.05, vega: 0.2, rho: 0.1 });

    await waitFor(() => expect(methodSelect()).not.toBeDisabled());
    expect(screen.getByLabelText("Stock Price (S):")).not.toBeDisabled();
  });
});

// --------------------------------------------------------------------------
// Error surfacing
// --------------------------------------------------------------------------

describe("a rejected request", () => {
  it("surfaces the API's error rather than failing silently", async () => {
    priceOption.mockRejectedValue(new Error("T must be a number"));
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /calculate price/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("T must be a number");
  });
});

// --------------------------------------------------------------------------
// Results render for every method
// --------------------------------------------------------------------------

describe("results", () => {
  const METHODS = ["BSM", "BT", "TT", "MC", "FD"];

  it.each(METHODS)("renders a full result for %s", async (method) => {
    priceOption.mockResolvedValue({ price: 12.34, delta: 0.5, gamma: 0.01, theta: -0.02, vega: 0.3, rho: 0.15 });
    render(<App />);
    await userEvent.selectOptions(methodSelect(), method);

    await userEvent.click(screen.getByRole("button", { name: /calculate price/i }));

    expect(await screen.findByText("12.340")).toBeInTheDocument();
    const call = priceOption.mock.calls.at(-1)[0];
    expect(call.method).toBe(method);
  });

  it("renders a full result for LSMC under American exercise", async () => {
    priceOption.mockResolvedValue({ price: 8.5, delta: 0.4, gamma: 0.02, theta: -0.03, vega: 0.25, rho: 0.1 });
    render(<App />);
    await userEvent.click(screen.getByLabelText("American"));
    await userEvent.selectOptions(methodSelect(), "LSMC");

    await userEvent.click(screen.getByRole("button", { name: /calculate price/i }));

    expect(await screen.findByText("8.500")).toBeInTheDocument();
    const call = priceOption.mock.calls.at(-1)[0];
    expect(call.method).toBe("LSMC");
    expect(call.exercise_type).toBe("american");
  });

  it("renders N/A rather than crashing on a null Greek", async () => {
    // The server returns null (never a bare NaN token) for a Greek that
    // couldn't be estimated at the edge of a method's numerical domain.
    priceOption.mockResolvedValue({ price: 5.0, delta: null, gamma: 0.01, theta: -0.02, vega: 0.3, rho: 0.15 });
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: /calculate price/i }));

    expect(await screen.findByText("5.000")).toBeInTheDocument();
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------
// URL-hash tab routing -- a view is linkable and survives a refresh
// --------------------------------------------------------------------------

describe("hash-based tab routing", () => {
  it("starts on the calculator when the hash is empty", () => {
    render(<App />);

    expect(screen.getByRole("tab", { name: "Calculator" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByTestId("analytics-view-stub")).not.toBeInTheDocument();
  });

  it("starts on analytics when loaded with #analytics", () => {
    window.location.hash = "#analytics";
    render(<App />);

    expect(screen.getByRole("tab", { name: "Analytics" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("analytics-view-stub")).toBeInTheDocument();
  });

  it("clicking the Analytics tab updates the URL hash", async () => {
    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: "Analytics" }));

    expect(window.location.hash).toBe("#analytics");
    expect(screen.getByTestId("analytics-view-stub")).toBeInTheDocument();
  });

  it("clicking back to Calculator clears the hash", async () => {
    window.location.hash = "#analytics";
    render(<App />);

    await userEvent.click(screen.getByRole("tab", { name: "Calculator" }));

    expect(window.location.hash).toBe("");
    expect(screen.queryByTestId("analytics-view-stub")).not.toBeInTheDocument();
  });

  it("a dispatched hashchange event switches the visible view (browser back/forward)", () => {
    render(<App />);
    expect(screen.queryByTestId("analytics-view-stub")).not.toBeInTheDocument();

    // A raw window event, unlike userEvent/fireEvent, isn't automatically
    // act()-wrapped -- do it explicitly rather than relying on waitFor to
    // paper over the resulting "not wrapped in act" warning.
    act(() => {
      window.location.hash = "#analytics";
      window.dispatchEvent(new HashChangeEvent("hashchange"));
    });

    expect(screen.getByTestId("analytics-view-stub")).toBeInTheDocument();
  });
});
