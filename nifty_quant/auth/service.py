"""Core authentication business logic: password hashing and JWT management.

Configuration is loaded from environment variables with safe dev defaults.
In production, always set ``SECRET_KEY`` to a strong random value.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

# ── Configuration ────────────────────────────────────────────────────────────

SECRET_KEY: str = os.environ.get(
    "SECRET_KEY",
    "dev-secret-key-do-not-use-in-production",
)
if os.environ.get("VERCEL") and SECRET_KEY == "dev-secret-key-do-not-use-in-production":
    raise RuntimeError("SECRET_KEY must be configured in Vercel environment variables")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


# ── Password hashing (bcrypt) ───────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Return a bcrypt hash of *password*."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check *plain_password* against a bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ── JWT tokens ───────────────────────────────────────────────────────────────


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Encode an access JWT containing *data* plus ``exp`` and ``type``."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": uuid.uuid4().hex,
        "type": "access",
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict[str, Any]) -> str:
    """Encode a long-lived refresh JWT."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "jti": uuid.uuid4().hex,
        "type": "refresh",
    })
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT.  Raises ``jwt.InvalidTokenError`` on failure."""
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ── Helpers ──────────────────────────────────────────────────────────────────


def hash_token(token: str) -> str:
    """SHA-256 hash of a raw token string (used for DB storage of refresh tokens)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_email(email: str) -> bool:
    """Basic structural email validation (not deliverability)."""
    return bool(_EMAIL_RE.match(email))
