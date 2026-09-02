// Thin wrapper around POST /api/price (api/pricing_routes.py). Same-origin in
// production since the built client is served by the same Flask host; the
// dev server proxies /api to Flask instead (vite.config.js).
export async function priceOption(payload) {
  const response = await fetch("/api/price", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    const message = body && body.error ? body.error : `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  if (body === null) {
    // A 200 whose body didn't parse as JSON is not a priced result -- an
    // unparseable success must surface as an error the caller can show,
    // not silently resolve as if the request had actually succeeded.
    throw new Error("The server returned an unreadable response.");
  }

  return body;
}
