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

    // One placeholder for the short skew curve (2026-12-18), one for the
    // short term-structure line (strike 100) -- both excluded curves, shown
    // rather than silently dropped.
    const placeholders = await screen.findAllByText(/Insufficient data/);
    expect(placeholders).toHaveLength(2);
    // The one OK curve (skew, 2026-09-18) still renders its chart alongside them.
    expect(screen.getAllByTestId("chart-point-implied_vol").length).toBe(3);
  });

  it("renders violations as a table with a visible count", async () => {
    render(<AnalyticsView />);

    expect(await screen.findByText("Violations (1)")).toBeInTheDocument();
    const table = screen.getByRole("table", { name: /recorded no-arbitrage violations/i });
    expect(within(table).getByText("put_call_band")).toBeInTheDocument();
    expect(within(table).getByText("120")).toBeInTheDocument();
  });
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
  await screen.findByText("Violations (1)");

  let resolveSurface, resolveComparison, resolveViolations;
  fetchSurface.mockReturnValue(new Promise((resolve) => (resolveSurface = resolve)));
  fetchComparison.mockReturnValue(new Promise((resolve) => (resolveComparison = resolve)));
  fetchViolations.mockReturnValue(new Promise((resolve) => (resolveViolations = resolve)));

  await userEvent.selectOptions(screen.getByLabelText("Symbol"), "QQQ");

  // The previous symbol's data is gone immediately, replaced by one loading state.
  expect(screen.queryByText("Violations (1)")).not.toBeInTheDocument();
  expect(screen.getByRole("status")).toHaveTextContent("Loading analytics for QQQ");

  resolveSurface(surfaceFor("QQQ"));
  resolveComparison(comparisonFor("QQQ"));
  // Still loading -- not every response has landed yet.
  await waitFor(() => expect(screen.getByRole("status")).toBeInTheDocument());

  resolveViolations(violationsFor("QQQ"));

  expect(await screen.findByText("Violations (1)")).toBeInTheDocument();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});
