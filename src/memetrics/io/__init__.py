"""Platform readers. Each reader produces the common items table (toolkit/DESIGN.md). Readers may do I/O and
network calls; metric code never does."""
from __future__ import annotations

import hashlib
import os

ITEM_COLUMNS = ["item_id", "ts", "author", "community", "kind", "text", "unit", "parent", "engagement"]


def hash_author(author: str, salt: str | None = None) -> str:
    """Salted SHA-256 (first 16 hex chars) of an author handle or DID. The salt defaults to $MEMETRICS_SALT;
    keep it private so hashes cannot be reversed by dictionary attack."""
    salt = salt if salt is not None else os.environ.get("MEMETRICS_SALT", "")
    return hashlib.sha256(f"{salt}:{author}".encode()).hexdigest()[:16]
