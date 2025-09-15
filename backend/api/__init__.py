"""
API endpoint modules
"""

from . import (
    auth,
    doors,
    events,
    health,
    people,
    settings,
    stats,
    streams,
    webhooks,
    zones
)

__all__ = [
    "auth",
    "doors",
    "events",
    "health",
    "people",
    "settings",
    "stats",
    "streams",
    "webhooks",
    "zones"
]