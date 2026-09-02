"""JSON API blueprints (KTD11): pricing, and the read-only analytics views.

    from api import analytics_bp, api_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(analytics_bp)

KTD11 keeps Flask rather than migrating frameworks for this — the surface is
a couple of blueprints, and a framework change would spend the schedule
without producing evidence a reviewer values. See ``pricing_routes`` for the
pricing request/response shape and R29's sizing bounds, and
``analytics_routes`` for the surface/comparison/violations views (R19-R21).
"""

from .analytics_routes import analytics_bp
from .pricing_routes import RateLimiter, ValidationError, api_bp

__all__ = ["RateLimiter", "ValidationError", "analytics_bp", "api_bp"]
