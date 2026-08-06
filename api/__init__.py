"""JSON API blueprint (KTD11): the same pricing engine, exposed with bounds.

    from api import api_bp
    app.register_blueprint(api_bp)

KTD11 keeps Flask rather than migrating frameworks for this — the surface is
one blueprint, and a framework change would spend the schedule without
producing evidence a reviewer values. See ``pricing_routes`` for the request
and response shape and R29's sizing bounds.
"""

from .pricing_routes import RateLimiter, ValidationError, api_bp

__all__ = ["RateLimiter", "ValidationError", "api_bp"]
