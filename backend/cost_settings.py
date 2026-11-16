"""Validated cost controls, independent of listing-detail freshness."""
import os


def seconds(name, default, minimum, maximum):
    value = os.getenv(name, str(default))
    if not value.isascii() or not value.isdecimal() or not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(value)


def completed_scan_reuse():
    return seconds("COMPLETED_SCAN_REUSE_SECONDS", 3600, 3600, 86400)


def peer_ttl():
    return seconds("RIA_PEER_CACHE_TTL_SECONDS", 3600, 1800, 3600)


def active_interval():
    return seconds("RIA_ACTIVE_WINDOW_INTERVAL_SECONDS", 300, 60, 86400)


def active_enabled():
    value = os.getenv("RIA_ACTIVE_WINDOW_ENABLED", "false").lower()
    if value not in {"true", "false"}:
        raise ValueError("RIA_ACTIVE_WINDOW_ENABLED must be true or false")
    return value == "true"


def validate():
    completed_scan_reuse(), peer_ttl(), active_interval(), active_enabled()
