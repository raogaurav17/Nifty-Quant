"""Database-backed auth models using Neon Postgres or local SQLite.

Neon supplies ``DATABASE_URL`` in production. SQLite remains available when
that variable is absent so local development and the test suite stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import sqlite3
from typing import Any, ClassVar
import uuid

from dotenv import load_dotenv

load_dotenv(".env.local")

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - available in production installs
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]


class DoesNotExist(LookupError):
    """Raised when an auth record cannot be found."""


class IntegrityError(RuntimeError):
    """Raised when a database uniqueness constraint is violated."""


class Database:
    """Small database adapter shared by the auth model operations."""

    def __init__(self) -> None:
        self._url: str | None = None
        self._sqlite_path: str | None = None
        self._initialized = False

    @property
    def database(self) -> str | None:
        return self._url or self._sqlite_path if self._initialized else None

    def init(self, *, url: str | None = None, sqlite_path: str | None = None) -> None:
        self._url = url
        self._sqlite_path = sqlite_path
        self._initialized = True

    def close(self) -> None:
        self._initialized = False

    def is_postgres(self) -> bool:
        return self._url is not None

    def connection(self) -> Any:
        if not self._initialized:
            init_db()
        if self._url:
            if psycopg is None:
                raise RuntimeError("psycopg is required when DATABASE_URL is configured")
            return psycopg.connect(self._url, row_factory=dict_row)
        if self._sqlite_path is None:
            raise RuntimeError("Database has not been initialized")
        connection = sqlite3.connect(self._sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def placeholder(self) -> str:
        return "%s" if self.is_postgres() else "?"


db = Database()


@dataclass
class User:
    id: str
    email: str
    username: str
    hashed_password: str
    is_active: bool
    created_at: Any

    DoesNotExist: ClassVar[type[DoesNotExist]] = DoesNotExist

    @classmethod
    def _from_row(cls, row: Any) -> "User":
        return cls(
            id=str(row["id"]),
            email=row["email"],
            username=row["username"],
            hashed_password=row["hashed_password"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
        )

    @classmethod
    def create(cls, *, email: str, username: str, hashed_password: str) -> "User":
        user_id = str(uuid.uuid4())
        ph = db.placeholder()
        query = f"""
            INSERT INTO users (id, email, username, hashed_password)
            VALUES ({ph}, {ph}, {ph}, {ph})
            RETURNING id, email, username, hashed_password, is_active, created_at
        """
        try:
            with db.connection() as connection:
                row = connection.execute(
                    query,
                    (user_id, email, username, hashed_password),
                ).fetchone()
                connection.commit()
        except Exception as exc:
            if _is_integrity_error(exc):
                raise IntegrityError(str(exc)) from exc
            raise
        if row is None:
            raise RuntimeError("User insert returned no row")
        return cls._from_row(row)

    @classmethod
    def get(cls, condition: Any) -> "User":
        field, value = condition
        if field not in {"email", "id"}:
            raise ValueError(f"Unsupported user lookup field: {field}")
        ph = db.placeholder()
        with db.connection() as connection:
            row = connection.execute(
                f"SELECT id, email, username, hashed_password, is_active, created_at "
                f"FROM users WHERE {field} = {ph}",
                (value,),
            ).fetchone()
        if row is None:
            raise cls.DoesNotExist
        return cls._from_row(row)

    @classmethod
    def get_by_id(cls, user_id: str) -> "User":
        return cls.get(("id", user_id))


@dataclass
class RefreshToken:
    id: str
    user_id: str
    token_hash: str
    expires_at: Any
    revoked: bool
    created_at: Any

    DoesNotExist: ClassVar[type[DoesNotExist]] = DoesNotExist

    @classmethod
    def create(
        cls,
        *,
        user: User,
        token_hash: str,
        expires_at: datetime,
    ) -> "RefreshToken":
        token_id = str(uuid.uuid4())
        ph = db.placeholder()
        query = f"""
            INSERT INTO refresh_tokens
                (id, user_id, token_hash, expires_at)
            VALUES ({ph}, {ph}, {ph}, {ph})
            RETURNING id, user_id, token_hash, expires_at, revoked, created_at
        """
        try:
            with db.connection() as connection:
                row = connection.execute(
                    query,
                    (token_id, user.id, token_hash, expires_at),
                ).fetchone()
                connection.commit()
        except Exception as exc:
            if _is_integrity_error(exc):
                raise IntegrityError(str(exc)) from exc
            raise
        if row is None:
            raise RuntimeError("Refresh token insert returned no row")
        return cls._from_row(row)

    @classmethod
    def _from_row(cls, row: Any) -> "RefreshToken":
        return cls(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            token_hash=row["token_hash"],
            expires_at=row["expires_at"],
            revoked=bool(row["revoked"]),
            created_at=row["created_at"],
        )

    @classmethod
    def get(cls, condition: Any) -> "RefreshToken":
        field, value = condition
        if field != "token_hash":
            raise ValueError(f"Unsupported refresh-token lookup field: {field}")
        ph = db.placeholder()
        with db.connection() as connection:
            row = connection.execute(
                f"SELECT id, user_id, token_hash, expires_at, revoked, created_at "
                f"FROM refresh_tokens WHERE token_hash = {ph}",
                (value,),
            ).fetchone()
        if row is None:
            raise cls.DoesNotExist
        return cls._from_row(row)

    def save(self) -> None:
        ph = db.placeholder()
        with db.connection() as connection:
            connection.execute(
                f"UPDATE refresh_tokens SET revoked = {ph} WHERE id = {ph}",
                (self.revoked, self.id),
            )
            connection.commit()


def _is_integrity_error(exc: Exception) -> bool:
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    return psycopg is not None and isinstance(exc, psycopg.errors.UniqueViolation)


def init_db(db_path: str | None = None) -> None:
    """Initialize Neon Postgres, or SQLite when an explicit path is supplied."""
    database_url = None if db_path else os.environ.get("DATABASE_URL")
    if database_url:
        db.init(url=database_url)
    else:
        path = Path(db_path or "data/nifty_quant.db")
        path.parent.mkdir(parents=True, exist_ok=True)
        db.init(sqlite_path=str(path))

    if db.is_postgres():
        id_type = "TEXT"
        timestamp_type = "TIMESTAMPTZ"
        boolean_type = "BOOLEAN"
    else:
        id_type = "TEXT"
        timestamp_type = "TEXT"
        boolean_type = "INTEGER"

    with db.connection() as connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS users (
                id {id_type} PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                username VARCHAR(100) NOT NULL UNIQUE,
                hashed_password VARCHAR(255) NOT NULL,
                is_active {boolean_type} NOT NULL DEFAULT TRUE,
                created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id {id_type} PRIMARY KEY,
                user_id {id_type} NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash VARCHAR(64) NOT NULL UNIQUE,
                expires_at {timestamp_type} NOT NULL,
                revoked {boolean_type} NOT NULL DEFAULT FALSE,
                created_at {timestamp_type} NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
