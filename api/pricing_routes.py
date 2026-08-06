"""The bounded JSON pricing endpoint (R22, R29, KTD11).

    POST /api/price
    {
        "exercise_type": "european" | "american",
        "method": "BSM" | "BT" | "TT" | "MC" | "LSMC" | "FD",
        "option_type": "call" | "put",
        "S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "y": 0.02,

        # method-dependent sizing, see the bound constants below
        "time_steps": 200,                       # BT, TT
        "iterations": 10000, "seed": 42,          # MC, LSMC
        "timestep": 0.004,                        # MC, FD
    }

Response is ``pricing.PriceResult``'s own fields — the uniform contract shape
U4 gave every method, unchanged by the trip through JSON:

    {"price": ..., "delta": ..., "gamma": ..., "theta": ..., "vega": ..., "rho": ...}

A validation failure returns 400 with ``{"error": "<reason>"}`` rather than a
stack trace; a client over the rate limit returns 429 the same way.

``app.py``'s form has parsed sizing fields straight into the library with no
ceiling on any of them since it was written. This endpoint is the bounded
surface meant for programmatic and browser-client callers instead, ahead of
U15's React client replacing the form entirely.
"""

from __future__ import annotations

import time

from flask import Blueprint, jsonify, request

from pricing import AMERICAN_METHODS, EUROPEAN_METHODS, American_Option, European_Option

api_bp = Blueprint("api", __name__, url_prefix="/api")

#: BT/TT step count. The form this endpoint is meant to replace defaults to
#: 300 with no ceiling; the evidence generator's own convergence grid runs at
#: 200. 1000 sits comfortably above both — enough headroom for a caller
#: wanting more precision than either reference, while keeping a single
#: request's tree backward-pass bounded rather than open-ended.
MAX_TREE_STEPS = 1000

#: Path count for Monte Carlo and LSMC alike. The evidence generator runs MC
#: at 50,000 paths and LSMC at 20,000; this doubles the heavier of the two —
#: room to ask for more precision than the published baseline, not an
#: unbounded array.
MAX_ITERATIONS = 100_000

#: Floor on ``dt`` for finite differences and Monte Carlo. Bounding iteration
#: count alone still leaves ``M = int(T / dt)`` — the finite-difference
#: timestep count, and the second dimension of the Monte Carlo path array —
#: free to grow without limit as dt shrinks. This floor bounds M from above
#: for both. The library's own default resolution is 1/252 (~0.00397), about
#: four times coarser than this floor, so it and every less aggressive
#: request are unaffected.
MIN_TIMESTEP = 0.001

#: Requests allowed per client per window (R29's per-client rate limit),
#: keyed on source IP since this endpoint has no notion of an authenticated
#: caller yet. A rate limit keyed on ``request.remote_addr`` is only as
#: trustworthy as the network path in front of it: behind a reverse proxy
#: that does not set ``remote_addr`` from a trusted forwarded-for header,
#: every client can appear as one address (under-counting) or a client could
#: forge one to evade its own count (over-trusting a spoofable header would
#: be worse). Neither is fixed here — it needs the actual deployment's proxy
#: chain to get right, not a guess.
RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_SECONDS = 60.0

#: A valid request body is a few hundred bytes at most — a handful of numbers
#: and short strings. Bounding it keeps an oversized-body request from
#: costing memory before validation ever runs, independent of every sizing
#: bound below.
MAX_REQUEST_BYTES = 16 * 1024


class ValidationError(Exception):
    """A required field was missing, non-numeric, or outside its bound."""


class RateLimiter:
    """Sliding-window per-client request counting.

    In-memory and per-process — correct for the single-process deployment
    this app runs under (KTD11 keeps Flask rather than taking on a second
    framework or a datastore for one counter). ``clock`` is injectable so
    tests can drive the window without a real sleep.
    """

    def __init__(self, *, max_requests: int, window_seconds: float, clock=time.time):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._hits: dict[str, list[float]] = {}

    def allow(self, client_id: str) -> bool:
        now = self._clock()
        cutoff = now - self._window_seconds
        hits = [t for t in self._hits.get(client_id, []) if t > cutoff]
        allowed = len(hits) < self._max_requests
        if allowed:
            hits.append(now)
        self._hits[client_id] = hits
        return allowed

    def reset(self) -> None:
        """Test-only: clear all recorded hits between cases."""
        self._hits.clear()


rate_limiter = RateLimiter(
    max_requests=RATE_LIMIT_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS
)


@api_bp.record_once
def _apply_body_size_limit(state) -> None:
    """Applied once, when this blueprint is registered onto an app.

    Not ``setdefault``: Flask's own default config already carries the key
    ``MAX_CONTENT_LENGTH: None``, so ``setdefault`` sees it as already set and
    never applies this blueprint's bound.
    """
    state.app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES


def _require(payload: dict, field: str):
    value = payload.get(field)
    if value is None:
        raise ValidationError(f"{field} is required")
    return value


def _require_float(payload: dict, field: str) -> float:
    value = _require(payload, field)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number") from None


def _require_int(payload: dict, field: str) -> int:
    value = _require(payload, field)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number") from None


def _require_choice(payload: dict, field: str, choices) -> str:
    value = _require(payload, field)
    if value not in choices:
        raise ValidationError(f"{field} must be one of {', '.join(choices)}")
    return value


def _bounded_int(payload: dict, field: str, *, maximum: int) -> int:
    value = _require_int(payload, field)
    if value < 1:
        raise ValidationError(f"{field} must be positive")
    if value > maximum:
        raise ValidationError(f"{field} exceeds the maximum of {maximum}")
    return value


def _bounded_timestep(payload: dict, field: str) -> float:
    value = _require_float(payload, field)
    if value < MIN_TIMESTEP:
        raise ValidationError(f"{field} is below the minimum of {MIN_TIMESTEP}")
    return value


def _price_from_payload(payload: dict) -> dict:
    exercise_type = _require_choice(payload, "exercise_type", ("european", "american"))
    valid_methods = EUROPEAN_METHODS if exercise_type == "european" else AMERICAN_METHODS
    method = _require_choice(payload, "method", valid_methods)
    option_type = _require_choice(payload, "option_type", ("call", "put"))

    S = _require_float(payload, "S")
    K = _require_float(payload, "K")
    T = _require_float(payload, "T")
    r = _require_float(payload, "r")
    sigma = _require_float(payload, "sigma")
    y = _require_float(payload, "y")

    option_cls = European_Option if exercise_type == "european" else American_Option
    option = option_cls(option_type, S, K, r, sigma, y, T, method)

    if method in ("BT", "TT"):
        option.setTreeSteps(_bounded_int(payload, "time_steps", maximum=MAX_TREE_STEPS))
    elif method == "MC":
        seed = _require_int(payload, "seed")
        n = _bounded_int(payload, "iterations", maximum=MAX_ITERATIONS)
        dt = _bounded_timestep(payload, "timestep")
        option.setSeedVariables(seed=seed, n=n, dt=dt)
    elif method == "LSMC":
        seed = _require_int(payload, "seed")
        n = _bounded_int(payload, "iterations", maximum=MAX_ITERATIONS)
        option.setSeedVariables(seed=seed, n=n)
    elif method == "FD":
        dt = _bounded_timestep(payload, "timestep")
        option.setFDResolution(dt=dt)

    result = option.priceOption()
    return {
        "price": result.price,
        "delta": result.delta,
        "gamma": result.gamma,
        "theta": result.theta,
        "vega": result.vega,
        "rho": result.rho,
    }


@api_bp.route("/price", methods=["POST"])
def price():
    client_id = request.remote_addr or "unknown"
    if not rate_limiter.allow(client_id):
        return jsonify(error="rate limit exceeded"), 429

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="request body must be JSON"), 400

    try:
        return jsonify(_price_from_payload(payload))
    except ValidationError as exc:
        return jsonify(error=str(exc)), 400
