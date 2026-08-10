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

``S``, ``K``, ``T``, and ``sigma`` must be finite and strictly positive; ``T``
is additionally capped at ``MAX_MATURITY_YEARS`` and ``sigma`` at
``pricing.implied.MAX_VOL``. ``r`` and ``y`` must be finite but may be
negative. ``MC``'s ``iterations x (T / timestep)`` and ``FD``'s
``T / timestep`` are each bounded directly (``MAX_MC_CELLS``,
``MAX_FD_STEPS``) since a per-field bound on ``timestep`` alone cannot bound
their product once ``T`` is in play.

Response is ``pricing.PriceResult``'s own fields — the uniform contract shape
U4 gave every method, unchanged by the trip through JSON except that a
non-finite Greek (still reachable at the edges of a method's own numerical
domain) becomes ``null`` rather than a bare NaN/Infinity token, which is not
valid JSON:

    {"price": ..., "delta": ..., "gamma": ..., "theta": ..., "vega": ..., "rho": ...}

A validation failure returns 400 with ``{"error": "<reason>"}`` rather than a
stack trace; a client over the rate limit returns 429 the same way; a body
over ``MAX_REQUEST_BYTES`` returns 413, also as ``{"error": "<reason>"}``.

``app.py``'s form has parsed sizing fields straight into the library with no
ceiling on any of them since it was written. This endpoint is the bounded
surface meant for programmatic and browser-client callers instead, ahead of
U15's React client replacing the form entirely.
"""

from __future__ import annotations

import math
import time

from flask import Blueprint, jsonify, request

from pricing import AMERICAN_METHODS, EUROPEAN_METHODS, American_Option, European_Option
from pricing.implied import MAX_VOL

api_bp = Blueprint("api", __name__, url_prefix="/api")

#: BT/TT step count. The form this endpoint is meant to replace defaults to
#: 300 with no ceiling; the evidence generator's own convergence grid runs at
#: 200. 1000 sits comfortably above both — enough headroom for a caller
#: wanting more precision than either reference, while keeping a single
#: request's tree backward-pass bounded rather than open-ended.
MAX_TREE_STEPS = 1000

#: Path count for Monte Carlo. The evidence generator runs MC at 50,000 paths
#: and 50 steps (2,500,000 cells); this ceiling on the raw field is a coarse
#: sanity bound. ``MAX_MC_CELLS`` below is what actually bounds the request's
#: cost, since ``iterations`` alone says nothing about the path-array's other
#: dimension.
MAX_ITERATIONS = 100_000

#: Path count for LSMC, held far below ``MAX_ITERATIONS``. LSMC's internal
#: step count is ``m = int(sqrt(n))`` (pricing/american.py), and its backward
#: pass runs an ``np.linalg.lstsq`` once per step — cost measured directly
#: (not estimated) grows worse than linearly in ``n``: 5,000 paths repriced
#: (9x, via priceOption's Greek bumps) took 0.47s, 10,000 took 1.56s, and
#: 20,000 — the evidence generator's own LSMC baseline — took 6.18s. The old
#: shared ``MAX_ITERATIONS`` of 100,000 measured in the multi-minute range for
#: a single request. This caps LSMC at exactly the evidence generator's own
#: reference count, where the measured cost is a few seconds.
MAX_LSMC_ITERATIONS = 20_000

#: Bounds Monte Carlo's total path-array size (``iterations x M``, where
#: ``M = int(T / timestep)``) directly, since bounding ``iterations`` and
#: ``timestep`` independently still lets their product grow without limit.
#: Set to the evidence generator's own MC grid (50,000 paths x 50 steps),
#: measured directly at 0.83s for a full ``priceOption()`` call (9x reprice
#: included) — the same reference the original field-level bounds cited but
#: never actually enforced as a product.
MAX_MC_CELLS = 2_500_000

#: Bounds the finite-difference step count (``M = int(round(T / timestep))``)
#: directly, for the same reason ``MAX_MC_CELLS`` bounds MC's product rather
#: than trusting the timestep floor alone. Measured directly: 1,000 steps
#: repriced (9x) took 0.93s; 10,000 steps (reachable today via a long T at the
#: timestep floor) took 9.0s. Matches ``MAX_TREE_STEPS`` for a consistent
#: per-request ceiling across the two step-counted methods.
MAX_FD_STEPS = 1000

#: Floor on ``dt`` for finite differences and Monte Carlo. Kept as a per-field
#: sanity bound; ``MAX_MC_CELLS`` and ``MAX_FD_STEPS`` are what actually bound
#: a request's cost, since a single floor on ``dt`` alone cannot — a long
#: enough ``T`` defeats it regardless of where the floor sits. The library's
#: own default resolution is 1/252 (~0.00397), about four times coarser than
#: this floor, so it and every less aggressive request are unaffected.
MIN_TIMESTEP = 0.001

#: Upper bound on ``T``, in years. Chosen well above anything this library is
#: tested against — the evidence generator's own convergence grid tops out at
#: T=2, and the finite-difference grid's measured accuracy envelope
#: (pricing/contracts.py) extends to T=3 — so it rejects only nonsensical
#: inputs (a zero, negative, or absurdly large maturity), not real contracts.
#: It does not, by itself, bound MC or FD cost; ``MAX_MC_CELLS`` and
#: ``MAX_FD_STEPS`` do that.
MAX_MATURITY_YEARS = 30.0

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
        self._evict_stale(cutoff)
        return allowed

    def _evict_stale(self, cutoff: float) -> None:
        """Drop every client whose entire recorded window has already expired.

        Filtering ``hits`` inside ``allow`` only ever prunes the *calling*
        client's own list — a client who simply stops calling never revisits
        its key, so that key would otherwise live in memory for the life of
        the process: one entry per distinct source IP ever seen, never freed.
        A full sweep only ever removes keys, so it keeps the dict bounded by
        recently-active clients instead of all-time ones. Cheap at this
        endpoint's scale (a single process, a per-client cap of
        ``RATE_LIMIT_REQUESTS`` every ``RATE_LIMIT_WINDOW_SECONDS``).
        """
        stale = [
            client_id for client_id, hits in self._hits.items()
            if not hits or max(hits) <= cutoff
        ]
        for client_id in stale:
            del self._hits[client_id]

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


def _finite_float(payload: dict, field: str) -> float:
    """A number, and not one of the two JSON cannot represent: NaN, Infinity.

    ``_require_float`` only checks that ``float()`` succeeds -- it succeeds
    on both, and a NaN or Infinity contract parameter reaches the pricing
    library and (depending on the method) either 500s or produces a result
    whose own fields are non-finite, which is not valid JSON either.
    """
    value = _require_float(payload, field)
    if not math.isfinite(value):
        raise ValidationError(f"{field} must be finite")
    return value


def _positive_float(payload: dict, field: str) -> float:
    value = _finite_float(payload, field)
    if value <= 0:
        raise ValidationError(f"{field} must be positive")
    return value


def _bounded_positive_float(payload: dict, field: str, *, maximum: float) -> float:
    value = _positive_float(payload, field)
    if value > maximum:
        raise ValidationError(f"{field} exceeds the maximum of {maximum}")
    return value


def _finite_result(value: float) -> float | None:
    """Non-finite Greeks (from a still-degenerate parameter combination, or a
    method whose Greek estimation breaks down at the edge of its domain) must
    not reach ``jsonify`` -- NaN and Infinity serialize as bare tokens that
    are not valid JSON, and a caller's ``response.json()`` then fails on a
    200. ``None`` is the honest "not a number" the wire format can hold."""
    return value if math.isfinite(value) else None


def _price_from_payload(payload: dict) -> dict:
    exercise_type = _require_choice(payload, "exercise_type", ("european", "american"))
    valid_methods = EUROPEAN_METHODS if exercise_type == "european" else AMERICAN_METHODS
    method = _require_choice(payload, "method", valid_methods)
    option_type = _require_choice(payload, "option_type", ("call", "put"))

    S = _positive_float(payload, "S")
    K = _positive_float(payload, "K")
    T = _bounded_positive_float(payload, "T", maximum=MAX_MATURITY_YEARS)
    r = _finite_float(payload, "r")
    sigma = _bounded_positive_float(payload, "sigma", maximum=MAX_VOL)
    y = _finite_float(payload, "y")

    option_cls = European_Option if exercise_type == "european" else American_Option
    option = option_cls(option_type, S, K, r, sigma, y, T, method)

    if method in ("BT", "TT"):
        option.setTreeSteps(_bounded_int(payload, "time_steps", maximum=MAX_TREE_STEPS))
    elif method == "MC":
        seed = _require_int(payload, "seed")
        n = _bounded_int(payload, "iterations", maximum=MAX_ITERATIONS)
        dt = _bounded_timestep(payload, "timestep")
        if n * max(1, int(T / dt)) > MAX_MC_CELLS:
            raise ValidationError(
                f"iterations x (T / timestep) exceeds the maximum of {MAX_MC_CELLS}"
            )
        option.setSeedVariables(seed=seed, n=n, dt=dt)
    elif method == "LSMC":
        seed = _require_int(payload, "seed")
        n = _bounded_int(payload, "iterations", maximum=MAX_LSMC_ITERATIONS)
        option.setSeedVariables(seed=seed, n=n)
    elif method == "FD":
        dt = _bounded_timestep(payload, "timestep")
        if int(round(T / dt)) > MAX_FD_STEPS:
            raise ValidationError(f"T / timestep exceeds the maximum of {MAX_FD_STEPS} steps")
        option.setFDResolution(dt=dt)

    result = option.priceOption()
    return {
        "price": _finite_result(result.price),
        "delta": _finite_result(result.delta),
        "gamma": _finite_result(result.gamma),
        "theta": _finite_result(result.theta),
        "vega": _finite_result(result.vega),
        "rho": _finite_result(result.rho),
    }


@api_bp.errorhandler(413)
def _request_too_large(_exc):
    """Same ``{"error": ...}`` envelope as every other rejection this route
    makes -- without this, exceeding ``MAX_REQUEST_BYTES`` falls through to
    Flask's default HTML error page, which the module's own documented
    response contract does not mention."""
    return jsonify(error="request body exceeds the maximum size"), 413


@api_bp.route("/price", methods=["POST"])
def price():
    client_id = request.remote_addr or "unknown"
    if not rate_limiter.allow(client_id):
        return jsonify(error="rate limit exceeded"), 429

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        # Catches both "not JSON at all" and "valid JSON that isn't an
        # object" (a bare list, number, or string) -- the latter reached
        # `_require`'s `payload.get(field)` and raised an unhandled
        # AttributeError instead of the documented 400.
        return jsonify(error="request body must be a JSON object"), 400

    try:
        return jsonify(_price_from_payload(payload))
    except ValidationError as exc:
        return jsonify(error=str(exc)), 400
