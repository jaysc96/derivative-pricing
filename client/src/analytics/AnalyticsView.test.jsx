import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AnalyticsView from "./AnalyticsView";
import { fetchComparison, fetchSurface, fetchSymbols, fetchViolations } from "./api";

vi.mock("./api", () => ({
  fetchSymbols: vi.fn(),
  fetchSurface: vi.fn(),
  fetchComparison: vi.fn(),
  fetchViolations: vi.fn(),
}));

// Surface3D mounts react-plotly.js, which calls Plotly.newPlot imperatively
// and tries to create a WebGL context -- jsdom has no GPU backend for that
// and hangs rather than failing fast. Surface3D.test.jsx covers its own
// trace/layout construction with a mocked Plot; these tests only need to
// know the panel is present, so a stub is both correct and the only way
// this file finishes in finite time.
vi.mock("./Surface3D", () => ({
  default: () => <div data-testid="surface-3d-stub" />,
}));

const SYMBOLS = { symbols: ["SPY", "QQQ"], default: "SPY" };

function surfaceFor(symbol) {
  return {
    symbol,
    capture_time: "2026-08-06T14:00:00+00:00",
    skew: [
      {
        expiry: "2026-09-18",
        option_type: "call",
        status: "ok",
        points: [
          { strike: 90, implied_vol: 0.2 },
          { strike: 100, implied_vol: 0.25 },
          { strike: 110, implied_vol: 0.3 },
        ],
      },
      { expiry: "2026-12-18", option_type: "call", status: "insufficient_data", points: [] },
    ],
    term_structure: [{ strike: 100, option_type: "call", status: "insufficient_data", points: [] }],
  };
}

function comparisonFor(symbol) {
  return {
    symbol,
    points: [
      { as_of: "2026-08-01", implied_vol: 0.22, realized_vol: null },
      { as_of: "2026-08-02", implied_vol: 0.23, realized_vol: 0.19 },
    ],
  };
}

function violationsFor(symbol) {
  return {
    symbol,
    count: 1,
    violations: [
      {
        expiry: "2026-09-18",
        strike: 120,
        option_type: "call",
        contract_symbol: "C120",
        kind: "put_call_band",
        detail: "call_ask(0.2) - put_bid(30) < lower bound",
      },
    ],
  };
}

function mockReadySymbol(symbol) {
  fetchSurface.mockResolvedValue(surfaceFor(symbol));
  fetchComparison.mockResolvedValue(comparisonFor(symbol));
  fetchViolations.mockResolvedValue(violationsFor(symbol));
}

const skewPanel = () => screen.getByRole("region", { name: "Skew" });
const violationsPanel = () => screen.getByRole("region", { name: "Violations" });

beforeEach(() => {
  fetchSymbols.mockReset();
  fetchSurface.mockReset();
  fetchComparison.mockReset();
  fetchViolations.mockReset();
});

// --------------------------------------------------------------------------
// The tracked set drives the symbol select (R37)
// --------------------------------------------------------------------------

it("populates the symbol select from the tracked set and selects the first on load", async () => {
  fetchSymbols.mockResolvedValue(SYMBOLS);
  mockReadySymbol("SPY");

  render(<AnalyticsView />);

  const select = await screen.findByLabelText("Symbol");
  expect(within(select).getAllByRole("option").map((o) => o.value)).toEqual(["SPY", "QQQ"]);
  expect(select).toHaveValue("SPY");
  await waitFor(() => expect(fetchSurface).toHaveBeenCalledWith("SPY"));
});

// --------------------------------------------------------------------------
// Rendered states
// --------------------------------------------------------------------------

describe("once a symbol's data has landed", () => {
  beforeEach(() => {
    fetchSymbols.mockResolvedValue(SYMBOLS);
    mockReadySymbol("SPY");
  });

  it("shows the snapshot's own capture timestamp", async () => {
    render(<AnalyticsView />);
    expect(await screen.findByText(/Snapshot captured/)).toBeInTheDocument();
  });

  it("renders an inline insufficient-data placeholder rather than an empty chart", async () => {
    render(<AnalyticsView />);

    // One for the short skew expiry (2026-12-18), one for the selected
    // strike's short term-structure line -- both shown in place, rather
    // than the affected curve being silently omitted.
    const placeholders = await screen.findAllByTestId("insufficient-data");
    expect(placeholders).toHaveLength(2);

    // The one OK skew curve still plots its three points alongside them.
    expect(within(skewPanel()).getAllByTestId("chart-point")).toHaveLength(3);
    expect(within(skewPanel()).getByText(/2026-12-18/)).toBeInTheDocument();
  });

  it("renders violations as a table with a visible count", async () => {
    render(<AnalyticsView />);

    const panel = await waitFor(violationsPanel);
    expect(within(panel).getByText("1 rejected leg")).toBeInTheDocument();

    const table = within(panel).getByRole("table", { name: /recorded no-arbitrage violations/i });
    expect(within(table).getByText("put_call_band")).toBeInTheDocument();
    expect(within(table).getByText("120")).toBeInTheDocument();
  });

  it("plots implied and realized as two separate series, skipping days a series lacks", async () => {
    render(<AnalyticsView />);

    const panel = await waitFor(() => screen.getByRole("region", { name: "Implied versus realized" }));
    const points = within(panel).getAllByTestId("chart-point");
    // Implied has both days; realized has only the second -- the null day is
    // skipped rather than plotted at zero.
    expect(points.filter((p) => p.dataset.series === "implied")).toHaveLength(2);
    expect(points.filter((p) => p.dataset.series === "realized")).toHaveLength(1);
  });

  it("passes the skew curves to the 3D surface panel and shows the plotted-expiry count", async () => {
    render(<AnalyticsView />);

    const panel = await waitFor(() => screen.getByRole("region", { name: "3D surface" }));
    expect(within(panel).getByTestId("surface-3d-stub")).toBeInTheDocument();
    // surfaceFor("SPY")'s skew has one curve with points and one with none --
    // the chip counts curves that actually have data to plot.
    expect(within(panel).getByText("1 expiries plotted")).toBeInTheDocument();
  });
});

// --------------------------------------------------------------------------
// Term structure: the strike selector defaults to a plottable strike
// --------------------------------------------------------------------------

it("defaults the term-structure strike selector to the first plottable strike, not the lowest", async () => {
  fetchSymbols.mockResolvedValue(SYMBOLS);
  fetchSurface.mockResolvedValue({
    symbol: "SPY",
    capture_time: "2026-08-06T14:00:00+00:00",
    skew: [],
    // The lowest strike (90) is thin -- exactly the low-wing shape that made
    // the old "default to strikes[0]" fallback open on an insufficient-data
    // notice almost every time. 100 is the first strike with a real curve.
    term_structure: [
      { strike: 90, option_type: "call", status: "insufficient_data", points: [] },
      {
        strike: 100,
        option_type: "call",
        status: "ok",
        points: [
          { expiry: "2026-09-18", implied_vol: 0.2 },
          { expiry: "2026-10-16", implied_vol: 0.22 },
          { expiry: "2026-11-20", implied_vol: 0.24 },
        ],
      },
    ],
  });
  fetchComparison.mockResolvedValue({ symbol: "SPY", points: [] });
  fetchViolations.mockResolvedValue({ symbol: "SPY", count: 0, violations: [] });

  render(<AnalyticsView />);

  const select = await screen.findByLabelText("Strike:");
  expect(select).toHaveValue("100");
  expect(screen.queryByTestId("insufficient-data")).not.toBeInTheDocument();
});

it("shows a no-data message rather than empty charts when the symbol has never been captured", async () => {
  fetchSymbols.mockResolvedValue(SYMBOLS);
  fetchSurface.mockResolvedValue({ symbol: "SPY", capture_time: null, skew: [], term_structure: [] });
  fetchComparison.mockResolvedValue({ symbol: "SPY", points: [] });
  fetchViolations.mockResolvedValue({ symbol: "SPY", count: 0, violations: [] });

  render(<AnalyticsView />);

  expect(await screen.findByText("Not yet captured for this symbol.")).toBeInTheDocument();
  expect(screen.getByText("No data captured yet for SPY.")).toBeInTheDocument();
});

it("surfaces a fetch error rather than failing silently", async () => {
  fetchSymbols.mockResolvedValue(SYMBOLS);
  fetchSurface.mockRejectedValue(new Error("symbol must be one of SPY, QQQ"));
  fetchComparison.mockResolvedValue(comparisonFor("SPY"));
  fetchViolations.mockResolvedValue(violationsFor("SPY"));

  render(<AnalyticsView />);

  expect(await screen.findByRole("alert")).toHaveTextContent("symbol must be one of SPY, QQQ");
});

// --------------------------------------------------------------------------
// Switching symbols: one combined loading state, previous charts cleared
// --------------------------------------------------------------------------

it("clears the previous symbol's charts and shows one loading state until all three requests land", async () => {
  fetchSymbols.mockResolvedValue(SYMBOLS);
  mockReadySymbol("SPY");

  render(<AnalyticsView />);
  await waitFor(violationsPanel);

  let resolveSurface, resolveComparison, resolveViolations;
  fetchSurface.mockReturnValue(new Promise((resolve) => (resolveSurface = resolve)));
  fetchComparison.mockReturnValue(new Promise((resolve) => (resolveComparison = resolve)));
  fetchViolations.mockReturnValue(new Promise((resolve) => (resolveViolations = resolve)));

  await userEvent.selectOptions(screen.getByLabelText("Symbol"), "QQQ");

  // The previous symbol's data is gone immediately, replaced by one loading state.
  expect(screen.queryByRole("region", { name: "Violations" })).not.toBeInTheDocument();
  expect(screen.queryByTestId("chart-point")).not.toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("Loading analytics for QQQ");

  resolveSurface(surfaceFor("QQQ"));
  resolveComparison(comparisonFor("QQQ"));
  // Still loading -- not every response has landed yet.
  await waitFor(() => expect(screen.getByRole("status")).toBeInTheDocument());

  resolveViolations(violationsFor("QQQ"));

  await waitFor(violationsPanel);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});
