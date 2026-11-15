"""FastAPI dependency-injection callables for authentication.

Two main dependencies:

* ``get_current_user``  — raises 401 if no valid token
* ``get_optional_user`` — returns ``None`` instead of raising
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
import jwt as pyjwt

from nifty_quant.auth.models import User
from nifty_quant.auth.service import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _extract_token(request: Request, bearer_token: str | None) -> str | None:
    """Extract JWT from ``Authorization: Bearer`` header or ``access_token`` cookie."""
    if bearer_token:
        return bearer_token
    return request.cookies.get("access_token")


def get_current_user(
    request: Request,
    bearer_token: str | None = Depends(oauth2_scheme),
) -> User:
    """Resolve the authenticated user from the request, or raise HTTP 401."""
    token = _extract_token(request, bearer_token)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(token)
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except pyjwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
        )

    try:
        user = User.get_by_id(user_id)
    except User.DoesNotExist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    return user


def get_optional_user(
    request: Request,
    bearer_token: str | None = Depends(oauth2_scheme),
) -> User | None:
    """Return the authenticated user if a valid token is present, else ``None``."""
    token = _extract_token(request, bearer_token)
    if token is None:
        return None
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            return None
        user_id = payload.get("sub")
        if user_id is None:
            return None
        user = User.get_by_id(user_id)
        return user if user.is_active else None
    except Exception:
        return None

