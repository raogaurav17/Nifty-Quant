"""Tests for the auth module: registration, login, token refresh, protected routes."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nifty_quant.auth.models import User, RefreshToken, init_db, db
from nifty_quant.auth.service import hash_password, verify_password, create_access_token, decode_token


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fresh_db(tmp_path: Path):
    """Ensure every test gets an empty in-memory database."""
    db_path = str(tmp_path / "test_auth.db")
    init_db(db_path)
    yield
    db.close()


# ── Service layer tests ──────────────────────────────────────────────────────


class TestPasswordHashing:
    def test_hash_and_verify(self) -> None:
        pw = "s3cureP@ssw0rd!"
        hashed = hash_password(pw)
        assert hashed != pw
        assert verify_password(pw, hashed)

    def test_wrong_password_fails(self) -> None:
        hashed = hash_password("correct")
        assert not verify_password("wrong", hashed)


class TestJWT:
    def test_create_and_decode(self) -> None:
        data = {"sub": "user-123", "email": "a@b.com"}
        token = create_access_token(data)
        payload = decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["email"] == "a@b.com"
        assert payload["type"] == "access"


# ── API tests (via TestClient) ───────────────────────────────────────────────


@pytest.fixture()
def client():
    from nifty_quant.web.app import app
    with TestClient(app) as c:
        yield c


class TestAuthAPI:
    def _register(self, client, email="test@example.com", username="testuser", password="password123"):
        return client.post("/auth/register", json={
            "email": email,
            "username": username,
            "password": password,
        })

    def test_register_success(self, client) -> None:
        resp = self._register(client)
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == "test@example.com"
        assert data["user"]["username"] == "testuser"
        # Cookies should be set
        assert "access_token" in resp.cookies
        assert "refresh_token" in resp.cookies

    def test_register_duplicate_email(self, client) -> None:
        self._register(client, email="dupe@example.com", username="user1")
        resp = self._register(client, email="dupe@example.com", username="user2")
        assert resp.status_code == 409

    def test_register_duplicate_username(self, client) -> None:
        self._register(client, email="a@example.com", username="sameuser")
        resp = self._register(client, email="b@example.com", username="sameuser")
        assert resp.status_code == 409

    def test_register_short_password(self, client) -> None:
        resp = self._register(client, password="short")
        assert resp.status_code == 422

    def test_register_invalid_email(self, client) -> None:
        resp = self._register(client, email="notanemail")
        assert resp.status_code == 422

    def test_login_success(self, client) -> None:
        self._register(client)
        resp = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["email"] == "test@example.com"

    def test_login_wrong_password(self, client) -> None:
        self._register(client)
        resp = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client) -> None:
        resp = client.post("/auth/login", json={
            "email": "nobody@example.com",
            "password": "password123",
        })
        assert resp.status_code == 401

    def test_me_authenticated(self, client) -> None:
        reg = self._register(client)
        token = reg.json()["access_token"]
        resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["email"] == "test@example.com"

    def test_me_no_token(self, client) -> None:
        resp = client.get("/auth/me")
        assert resp.status_code == 401

    def test_me_via_cookie(self, client) -> None:
        self._register(client)
        # Login sets cookies
        login_resp = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        # TestClient carries cookies forward automatically
        resp = client.get("/auth/me")
        assert resp.status_code == 200

    def test_refresh_rotates_tokens(self, client) -> None:
        self._register(client)
        login_resp = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        old_rt = login_resp.json()["refresh_token"]

        refresh_resp = client.post("/auth/refresh")
        assert refresh_resp.status_code == 200
        new_data = refresh_resp.json()
        assert new_data["refresh_token"] != old_rt

    def test_logout_clears_cookies(self, client) -> None:
        self._register(client)
        client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "password123",
        })
        logout_resp = client.post("/auth/logout")
        assert logout_resp.status_code == 204

        # After logout, /auth/me should fail
        me_resp = client.get("/auth/me")
        assert me_resp.status_code == 401

    def test_dashboard_redirects_when_unauthenticated(self, client) -> None:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 302
        assert "/login" in resp.headers["location"]

    def test_login_page_accessible(self, client) -> None:
        resp = client.get("/login")
        assert resp.status_code == 200
        assert "Sign In" in resp.text

