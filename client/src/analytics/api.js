// Thin wrappers around GET /api/analytics/* (api/analytics_routes.py). Same
// error-envelope convention as ../api.js: a non-2xx or unparseable body
// becomes a thrown Error carrying the server's own message where there is one.
async function getJSON(url) {
  const response = await fetch(url);
  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const message = body && body.error ? body.error : `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  if (body === null) {
    throw new Error("The server returned an unreadable response.");
  }

  return body;
}

export function fetchSymbols() {
  return getJSON("/api/analytics/symbols");
}

export function fetchSurface(symbol) {
  return getJSON(`/api/analytics/surface?symbol=${encodeURIComponent(symbol)}`);
}

export function fetchComparison(symbol) {
  return getJSON(`/api/analytics/comparison?symbol=${encodeURIComponent(symbol)}`);
}

export function fetchViolations(symbol) {
  return getJSON(`/api/analytics/violations?symbol=${encodeURIComponent(symbol)}`);
}
