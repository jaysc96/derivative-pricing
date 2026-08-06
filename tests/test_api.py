"""U14: the JSON pricing endpoint enforces the bounds `app.py`'s form never had.

Every test drives the endpoint through Flask's test client rather than calling
``_price_from_payload`` directly, so what is proven is the HTTP contract (status
code, JSON error shape) a real caller sees — not just that the validation
function raises the right exception internally.
"""

import sys
from pathlib import Path

import pytest

from api import pricing_routes

# app.py is deployment glue, not part of the installed package (pyproject.toml
# ships only pricing, marketdata, and api) — reached the same way
# scripts/generate_evidence.py reaches tests/reference_values.py.
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import app as flask_app  # noqa: E402


@pytest.fixture
def client():
    pricing_routes.rate_limiter.reset()
    flask_app.config.update(TESTING=True)
    with flask_app.test_client() as test_client:
        yield test_client
    pricing_routes.rate_limiter.reset()


def base_payload(**overrides):
    payload = {
        "exercise_type": "european",
        "method": "BSM",
        "option_type": "call",
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "y": 0.02,
    }
    payload.update(overrides)
    return payload


def post(client, payload):
    return client.post("/api/price", json=payload)


# --------------------------------------------------------------------------
# Every method is reachable and returns the uniform contract shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["BSM", "BT", "TT", "MC", "FD"])
def test_every_european_method_is_reachable(client, method):
    payload = base_payload(method=method)
    if method in ("BT", "TT"):
        payload["time_steps"] = 200
    elif method == "MC":
        payload.update(iterations=1000, seed=42, timestep=0.02)
    elif method == "FD":
        payload["timestep"] = 0.01

    response = post(client, payload)

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"price", "delta", "gamma", "theta", "vega", "rho"}
    assert all(isinstance(body[field], float) for field in body)
    assert body["price"] > 0


@pytest.mark.parametrize("method", ["BT", "TT", "LSMC", "FD"])
def test_every_american_method_is_reachable(client, method):
    payload = base_payload(exercise_type="american", method=method, option_type="put")
    if method in ("BT", "TT"):
        payload["time_steps"] = 200
    elif method == "LSMC":
        payload.update(iterations=1000, seed=42)
    elif method == "FD":
        payload["timestep"] = 0.01

    response = post(client, payload)

    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {"price", "delta", "gamma", "theta", "vega", "rho"}


def test_a_method_not_valid_for_the_exercise_type_is_rejected(client):
    """LSMC has no European implementation; BSM has no American one."""
    response = post(client, base_payload(exercise_type="european", method="LSMC"))
    assert response.status_code == 400
    assert "method" in response.get_json()["error"]


# --------------------------------------------------------------------------
# Sizing bounds
# --------------------------------------------------------------------------


def test_a_tree_step_count_above_the_bound_is_rejected(client):
    payload = base_payload(method="BT", time_steps=pricing_routes.MAX_TREE_STEPS + 1)
    response = post(client, payload)
    assert response.status_code == 400
    assert "time_steps" in response.get_json()["error"]


def test_a_tree_step_count_at_the_bound_is_accepted(client):
    payload = base_payload(method="BT", time_steps=pricing_routes.MAX_TREE_STEPS)
    assert post(client, payload).status_code == 200


def test_an_iteration_count_above_the_bound_is_rejected(client):
    payload = base_payload(
        method="MC",
        iterations=pricing_routes.MAX_ITERATIONS + 1,
        seed=42,
        timestep=0.02,
    )
    response = post(client, payload)
    assert response.status_code == 400
    assert "iterations" in response.get_json()["error"]


def test_an_lsmc_iteration_count_above_the_bound_is_rejected(client):
    payload = base_payload(
        exercise_type="american",
        method="LSMC",
        iterations=pricing_routes.MAX_ITERATIONS + 1,
        seed=42,
    )
    response = post(client, payload)
    assert response.status_code == 400
    assert "iterations" in response.get_json()["error"]


def test_a_timestep_below_the_floor_is_rejected_for_finite_differences(client):
    payload = base_payload(method="FD", timestep=pricing_routes.MIN_TIMESTEP / 2)
    response = post(client, payload)
    assert response.status_code == 400
    assert "timestep" in response.get_json()["error"]


def test_a_timestep_below_the_floor_is_rejected_for_monte_carlo(client):
    payload = base_payload(method="MC", iterations=1000, seed=42, timestep=pricing_routes.MIN_TIMESTEP / 2)
    response = post(client, payload)
    assert response.status_code == 400
    assert "timestep" in response.get_json()["error"]


def test_a_timestep_at_the_floor_is_accepted(client):
    payload = base_payload(method="FD", timestep=pricing_routes.MIN_TIMESTEP)
    assert post(client, payload).status_code == 200


# --------------------------------------------------------------------------
# Missing and non-numeric parameters
# --------------------------------------------------------------------------


def test_a_missing_required_parameter_is_rejected_not_raised(client):
    payload = base_payload()
    del payload["K"]
    response = post(client, payload)
    assert response.status_code == 400
    assert "K" in response.get_json()["error"]


def test_a_non_numeric_parameter_is_rejected_not_raised(client):
    response = post(client, base_payload(S="not-a-number"))
    assert response.status_code == 400
    assert "S" in response.get_json()["error"]


def test_a_non_numeric_sizing_parameter_is_rejected_not_raised(client):
    payload = base_payload(method="BT", time_steps="a lot")
    response = post(client, payload)
    assert response.status_code == 400
    assert "time_steps" in response.get_json()["error"]


def test_a_non_json_body_is_rejected_not_raised(client):
    response = client.post("/api/price", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_an_oversized_body_is_rejected_before_it_is_parsed(client):
    """MAX_CONTENT_LENGTH, not a sizing bound — a body-size DoS never reaches validation."""
    oversized = base_payload(padding="x" * pricing_routes.MAX_REQUEST_BYTES)
    response = post(client, oversized)
    assert response.status_code == 413


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


def test_requests_beyond_the_rate_limit_are_refused(client):
    payload = base_payload()
    for _ in range(pricing_routes.RATE_LIMIT_REQUESTS):
        assert post(client, payload).status_code == 200

    response = post(client, payload)
    assert response.status_code == 429
    assert "rate limit" in response.get_json()["error"]


def test_the_rate_limit_is_keyed_per_client(client):
    """A second client's requests are not consumed by the first's."""
    payload = base_payload()
    for _ in range(pricing_routes.RATE_LIMIT_REQUESTS):
        assert client.post("/api/price", json=payload, environ_overrides={"REMOTE_ADDR": "1.1.1.1"}).status_code == 200

    exhausted = client.post("/api/price", json=payload, environ_overrides={"REMOTE_ADDR": "1.1.1.1"})
    assert exhausted.status_code == 429

    fresh = client.post("/api/price", json=payload, environ_overrides={"REMOTE_ADDR": "2.2.2.2"})
    assert fresh.status_code == 200


def test_the_rate_limit_resets_outside_its_window():
    clock = {"now": 0.0}
    limiter = pricing_routes.RateLimiter(
        max_requests=2, window_seconds=60.0, clock=lambda: clock["now"]
    )
    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False

    clock["now"] += 61.0
    assert limiter.allow("client") is True
