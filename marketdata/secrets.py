"""Credential loading, built before anything needs it.

The current provider needs no credential, so R28 binds nothing today. That is
exactly why this exists now: the fallback adapter gets built under a trigger,
at the moment time pressure is highest, and reaching for an established pattern
then is very different from inventing one. A helper that reads the environment,
a documented variable name, and `.env` in `.gitignore` is small enough to build
cold and precisely the thing that should not be improvised hot.

Credentials are read from the environment at runtime. Never committed, never
baked into an image.
"""

from __future__ import annotations

import os
from pathlib import Path


class MissingCredential(RuntimeError):
    """A required credential was not set. Raised at startup, not at request time."""


def load_dotenv(path: str | Path = ".env") -> None:
    """Read `KEY=value` lines into the environment if the file exists.

    Never overwrites a variable already set — the real environment wins over a
    developer's local file, so a host's configured secret is not shadowed by a
    stale checkout.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Read a credential from the environment.

    Fails fast and by name when a required one is absent. The alternative — a
    None threaded through to an HTTP call — surfaces as an authentication error
    from the provider hours later, in a scheduled run nobody is watching.
    """
    value = os.environ.get(name)
    if value:
        return value
    if required:
        raise MissingCredential(
            f"{name} is not set. Export it, or add it to a local .env file "
            f"(gitignored). See the README for the variables each provider needs."
        )
    return None
