"""User and refresh-token models backed by SQLite via Peewee.

Tables are created lazily on first call to ``init_db()``.  The database file
lives in ``data/nifty_quant.db`` by default and is excluded from version control
via ``.gitignore``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import peewee as pw

db = pw.SqliteDatabase(None)  # deferred initialisation


# ── Base ─────────────────────────────────────────────────────────────────────


class BaseModel(pw.Model):
    """Base model binding all auth tables to the deferred SQLite database."""

    class Meta:
        database = db


# ── User ─────────────────────────────────────────────────────────────────────


class User(BaseModel):
    """Application user account."""

    id = pw.UUIDField(primary_key=True, default=uuid.uuid4)
    email = pw.CharField(unique=True, max_length=255, index=True)
    username = pw.CharField(unique=True, max_length=100, index=True)
    hashed_password = pw.CharField(max_length=255)
    is_active = pw.BooleanField(default=True)
    created_at = pw.DateTimeField(default=lambda: datetime.now(timezone.utc))

    class Meta:
        table_name = "users"


# ── Refresh Token ────────────────────────────────────────────────────────────


class RefreshToken(BaseModel):
    """Stores SHA-256 hashes of issued refresh tokens for server-side revocation.

    On each ``/auth/refresh`` call the old token is revoked and a new one is
    issued (token rotation).
    """

    id = pw.UUIDField(primary_key=True, default=uuid.uuid4)
    user = pw.ForeignKeyField(User, backref="refresh_tokens", on_delete="CASCADE")
    token_hash = pw.CharField(max_length=64, unique=True, index=True)
    expires_at = pw.DateTimeField()
    revoked = pw.BooleanField(default=False)
    created_at = pw.DateTimeField(default=lambda: datetime.now(timezone.utc))

    class Meta:
        table_name = "refresh_tokens"


# ── Initialisation ───────────────────────────────────────────────────────────


def init_db(db_path: str = "data/nifty_quant.db") -> None:
    """Open (or create) the SQLite database and ensure all tables exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db.init(
        str(path),
        pragmas={
            "journal_mode": "wal",
            "cache_size": -64_000,   # 64 MB
            "foreign_keys": 1,
            "busy_timeout": 5000,
        },
    )
    db.connect(reuse_if_open=True)
    db.create_tables([User, RefreshToken])

