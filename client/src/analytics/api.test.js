import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchComparison, fetchSurface, fetchSymbols, fetchViolations } from "./api";

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("fetchSymbols", () => {
  it("GETs the symbols endpoint and returns the parsed body", async () => {
    fetch.mockResolvedValue(jsonResponse({ symbols: ["SPY", "QQQ"], default: "SPY" }));

    const body = await fetchSymbols();

    expect(fetch).toHaveBeenCalledWith("/api/analytics/symbols");
    expect(body).toEqual({ symbols: ["SPY", "QQQ"], default: "SPY" });
  });
});

describe("the symbol-scoped endpoints", () => {
  it("fetchSurface encodes the symbol into the query string", async () => {
    fetch.mockResolvedValue(jsonResponse({ symbol: "SPY", capture_time: null, skew: [], term_structure: [] }));
    await fetchSurface("SPY");
    expect(fetch).toHaveBeenCalledWith("/api/analytics/surface?symbol=SPY");
  });

  it("fetchComparison encodes the symbol into the query string", async () => {
    fetch.mockResolvedValue(jsonResponse({ symbol: "SPY", points: [] }));
    await fetchComparison("SPY");
    expect(fetch).toHaveBeenCalledWith("/api/analytics/comparison?symbol=SPY");
  });

  it("fetchViolations encodes the symbol into the query string", async () => {
    fetch.mockResolvedValue(jsonResponse({ symbol: "SPY", count: 0, violations: [] }));
    await fetchViolations("SPY");
    expect(fetch).toHaveBeenCalledWith("/api/analytics/violations?symbol=SPY");
  });
});

describe("error handling", () => {
  it("surfaces the server's error message on a non-2xx response", async () => {
    fetch.mockResolvedValue(jsonResponse({ error: "symbol must be one of SPY, QQQ" }, { ok: false, status: 400 }));

    await expect(fetchSurface("ZZZZ")).rejects.toThrow("symbol must be one of SPY, QQQ");
  });

  it("falls back to a status-code message when the error body has no error field", async () => {
    fetch.mockResolvedValue(jsonResponse({}, { ok: false, status: 500 }));

    await expect(fetchSurface("SPY")).rejects.toThrow("Request failed with status 500");
  });

  it("rejects when a 200 response body doesn't parse as JSON", async () => {
    fetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.reject(new Error("not json")),
    });

    await expect(fetchSurface("SPY")).rejects.toThrow("The server returned an unreadable response.");
  });
});
