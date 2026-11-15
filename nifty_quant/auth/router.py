"""Auth API routes: register, login, token refresh, profile, and logout.

Tokens are delivered both as JSON (for API consumers) and as ``HttpOnly``
cookies (for the Jinja2 web dashboard).  Refresh tokens are stored as
SHA-256 hashes in the database and rotated on every refresh call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
import jwt as pyjwt
from nifty_quant.auth.dependencies import get_current_user
from nifty_quant.auth.models import IntegrityError, RefreshToken, User
from nifty_quant.auth.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from nifty_quant.auth.service import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    validate_email,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=str(user.id),
        email=user.email,
        username=user.username,
        is_active=user.is_active,
        created_at=str(user.created_at),
    )


def _set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
) -> None:
    """Write ``HttpOnly`` cookies for the web dashboard."""
    secure = bool(os.environ.get("VERCEL"))
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=secure,
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        samesite="lax",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        secure=secure,
    )


def _issue_tokens(user: User, response: Response) -> TokenResponse:
    """Create access + refresh tokens, persist the refresh hash, set cookies."""
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "username": user.username,
    }
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    RefreshToken.create(
        user=user,
        token_hash=hash_token(refresh_token),
        expires_at=datetime.now(timezone.utc)
        + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )

    _set_auth_cookies(response, access_token, refresh_token)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=_user_response(user),
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
def register(body: RegisterRequest, response: Response) -> TokenResponse:
    """Register a new user and return access + refresh tokens."""
    if not validate_email(body.email):
        raise HTTPException(status_code=422, detail="Invalid email format")

    try:
        user = User.create(
            email=body.email.lower().strip(),
            username=body.username.strip(),
            hashed_password=hash_password(body.password),
        )
    except IntegrityError as exc:
        detail = "Email or username already taken"
        err_msg = str(exc).lower()
        if "email" in err_msg:
            detail = "Email already registered"
        elif "username" in err_msg:
            detail = "Username already taken"
        raise HTTPException(status_code=409, detail=detail) from exc

    return _issue_tokens(user, response)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and obtain tokens",
)
def login(body: LoginRequest, response: Response) -> TokenResponse:
    """Validate credentials and return access + refresh tokens."""
    try:
        user = User.get(("email", body.email.lower().strip()))
    except User.DoesNotExist:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    return _issue_tokens(user, response)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate tokens using a valid refresh token",
)
def refresh(request: Request, response: Response) -> TokenResponse:
    """Consume the current refresh token and issue a fresh pair (rotation)."""
    token = request.cookies.get("refresh_token")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    if not token:
        raise HTTPException(status_code=401, detail="Refresh token required")

    try:
        payload = decode_token(token)
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired — please log in again")
    except pyjwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    # Server-side revocation check
    th = hash_token(token)
    try:
        stored = RefreshToken.get(("token_hash", th))
    except RefreshToken.DoesNotExist:
        raise HTTPException(status_code=401, detail="Refresh token not recognised")

    if stored.revoked:
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")

    # Rotate: revoke old token, issue new pair
    stored.revoked = True
    stored.save()

    user_id = payload.get("sub")
    try:
        user = User.get_by_id(user_id)
    except User.DoesNotExist:
        raise HTTPException(status_code=401, detail="User not found")

    return _issue_tokens(user, response)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Return the authenticated user's profile."""
    return _user_response(user)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out and revoke refresh token",
)
def logout(request: Request, response: Response) -> None:
    """Clear auth cookies and revoke the refresh token."""
    rt = request.cookies.get("refresh_token")
    if rt:
        th = hash_token(rt)
        try:
            stored = RefreshToken.get(("token_hash", th))
            stored.revoked = True
            stored.save()
        except RefreshToken.DoesNotExist:
            pass

    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
