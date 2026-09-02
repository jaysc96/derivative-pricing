import { afterEach, describe, expect, it, vi } from "vitest";
import { priceOption } from "./api";

// The one file in this codebase that actually exercises `fetch` and response
// parsing -- App.test.jsx mocks this whole module, so without these tests
// nothing drives the real request/response boundary.

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
  };
}

function unparseableResponse({ ok = true, status = 200 } = {}) {
  return {
    ok,
    status,
    json: () => Promise.reject(new SyntaxError("Unexpected token")),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("priceOption", () => {
  it("posts JSON to /api/price with the payload as the body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ price: 1.0 }));
    vi.stubGlobal("fetch", fetchMock);

    await priceOption({ method: "BSM", S: 100 });

    expect(fetchMock).toHaveBeenCalledWith("/api/price", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ method: "BSM", S: 100 }),
    });
  });

  it("returns the parsed body on a successful response", async () => {
    const body = { price: 12.34, delta: 0.5, gamma: 0.1, theta: -0.05, vega: 0.2, rho: 0.1 };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(body)));

    await expect(priceOption({})).resolves.toEqual(body);
  });

  it("throws the server's error message on a non-ok JSON response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ error: "T must be a number" }, { ok: false, status: 400 })),
    );

    await expect(priceOption({})).rejects.toThrow("T must be a number");
  });

  it("falls back to a status-based message when a non-ok response carries no JSON body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(unparseableResponse({ ok: false, status: 413 })));

    await expect(priceOption({})).rejects.toThrow("Request failed with status 413");
  });

  it("throws rather than resolving null when a 200 response body does not parse as JSON", async () => {
    // The exact shape a server bug that emits a bare NaN/Infinity token
    // produces: response.ok is true, but .json() rejects.
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(unparseableResponse({ ok: true, status: 200 })));

    await expect(priceOption({})).rejects.toThrow("unreadable response");
  });
});
