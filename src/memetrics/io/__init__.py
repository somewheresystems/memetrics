"""Platform readers. Each reader turns platform data into the items table (:mod:`memetrics.schema`).

Readers may read files and, where documented, make network requests; measure code never does. Network code lives
only in this package, is read-only, needs no login, identifies itself with a User-Agent, and honours the platform's
rate-limit headers.
"""

from __future__ import annotations

import hashlib
import os
import warnings

from .. import __version__

USER_AGENT = f"memetrics/{__version__} (research; read-only; +https://github.com/somewheresystems/memetrics)"


def hash_author(author: str, salt: str | None = None) -> str:
    """Salted SHA-256 (first 16 hex chars) of an author handle, DID, domain or wallet.

    ``salt`` defaults to ``$MEMETRICS_SALT``. Keep it private and fixed for a study: without a salt, hashes of known
    handles can be reversed by dictionary attack, so an empty salt triggers a warning.
    """
    salt = salt if salt is not None else os.environ.get("MEMETRICS_SALT", "")
    if not salt:
        warnings.warn("hash_author called without a salt; set MEMETRICS_SALT or pass salt=", stacklevel=2)
    return hashlib.sha256(f"{salt}:{author}".encode()).hexdigest()[:16]
