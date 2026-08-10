"""Read-only JSON views over the derived analytics layer (R19-R21, R33, R36, R37).

    GET /api/analytics/symbols
    GET /api/analytics/surface?symbol=SPY
    GET /api/analytics/comparison?symbol=SPY
    GET /api/analytics/violations?symbol=SPY

Every route reads the archive through ``analytics.surface``/``analytics.comparison``
and never the solver (KTD9) — a page load costs rows returned, not pricer
calls. ``symbol`` is checked against ``marketdata.TRACKED`` before any query
runs: R37 has the client select from that same set, and this is what keeps a
direct call with an arbitrary symbol from ever reaching the query layer,
per U18's approach.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from analytics.comparison import build_implied_vs_realized
from analytics.surface import build_skew, build_term_structure, build_violations
from marketdata import TRACKED, Store

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")

#: Matches scripts/capture.py's own default -- the archive this blueprint
#: reads is the one the scheduled job writes.
DEFAULT_DB = Path(__file__).parent.parent / "data" / "quotes.db"


class ValidationError(Exception):
    """The requested symbol is not one the capture job tracks."""


def _store() -> Store:
    """The archive this request reads, lazily opened on first use.

    Reading ``app.config`` at request time rather than opening a ``Store`` at
    import or blueprint-registration time is what lets a test inject its own
    archive (``app.config["ANALYTICS_STORE"] = Store(tmp_path / ...)``)
    without this module ever touching the real ``data/quotes.db`` — importing
    this module, or even registering the blueprint, must not create or
    migrate a database file a test never asked for.
    """
    store = current_app.config.get("ANALYTICS_STORE")
    if store is None:
        store = Store(DEFAULT_DB)
        current_app.config["ANALYTICS_STORE"] = store
    return store


def _validated_symbol() -> str:
    symbol = (request.args.get("symbol") or "").strip().upper()
    if symbol not in TRACKED:
        raise ValidationError(f"symbol must be one of {', '.join(TRACKED)}")
    return symbol


def _skew_json(curves):
    return [
        {
            "expiry": curve.expiry.isoformat(),
            "option_type": curve.option_type,
            "status": curve.status,
            "points": [{"strike": p.strike, "implied_vol": p.implied_vol} for p in curve.points],
        }
        for curve in curves
    ]


def _term_structure_json(curves):
    return [
        {
            "strike": curve.strike,
            "option_type": curve.option_type,
            "status": curve.status,
            "points": [{"expiry": p.expiry.isoformat(), "implied_vol": p.implied_vol} for p in curve.points],
        }
        for curve in curves
    ]


@analytics_bp.errorhandler(ValidationError)
def _invalid(exc):
    return jsonify(error=str(exc)), 400


@analytics_bp.route("/symbols")
def symbols():
    return jsonify(symbols=list(TRACKED), default=TRACKED[0])


@analytics_bp.route("/surface")
def surface():
    symbol = _validated_symbol()
    store = _store()
    capture_time = store.latest_capture(symbol)
    return jsonify(
        symbol=symbol,
        capture_time=capture_time.isoformat() if capture_time else None,
        skew=_skew_json(build_skew(store, symbol)),
        term_structure=_term_structure_json(build_term_structure(store, symbol)),
    )


@analytics_bp.route("/comparison")
def comparison():
    symbol = _validated_symbol()
    store = _store()
    points = build_implied_vs_realized(store, symbol)
    return jsonify(
        symbol=symbol,
        points=[
            {"as_of": p.as_of.isoformat(), "implied_vol": p.implied_vol, "realized_vol": p.realized_vol}
            for p in points
        ],
    )


@analytics_bp.route("/violations")
def violations():
    symbol = _validated_symbol()
    store = _store()
    rows = build_violations(store, symbol)
    return jsonify(
        symbol=symbol,
        count=len(rows),
        violations=[
            {
                "expiry": row.expiry.isoformat(),
                "strike": row.strike,
                "option_type": row.option_type,
                "contract_symbol": row.contract_symbol,
                "kind": row.kind,
                "detail": row.detail,
            }
            for row in rows
        ],
    )
