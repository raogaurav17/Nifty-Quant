"""Pydantic request / response schemas for the auth API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """Body for ``POST /auth/register``."""

    email: str = Field(..., min_length=5, max_length=255, examples=["user@example.com"])
    username: str = Field(
        ...,
        min_length=3,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        examples=["john_doe"],
    )
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    """Body for ``POST /auth/login``."""

    email: str = Field(..., examples=["user@example.com"])
    password: str = Field(...)


class UserResponse(BaseModel):
    """Public view of a user account (never exposes hashed_password)."""

    id: str
    email: str
    username: str
    is_active: bool
    created_at: str


class TokenResponse(BaseModel):
    """Returned after successful login or registration."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: UserResponse

