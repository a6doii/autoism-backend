from tests.conftest import register, login


def test_me_unauthenticated(client):
    r = client.get("/api/me")
    assert r.status_code == 200
    assert r.get_json() == {"authenticated": False, "user": None}


def test_register_success(client):
    r = register(client)
    assert r.status_code == 200
    body = r.get_json()
    assert body["user"]["email"] == "user@example.com"
    assert body["user"]["is_admin"] is False


def test_register_logs_in_immediately(client):
    register(client)
    r = client.get("/api/me")
    assert r.get_json()["authenticated"] is True


def test_register_duplicate_email(client):
    register(client)
    r = register(client)
    assert r.status_code == 400
    assert "already registered" in r.get_json()["error"]


def test_register_password_mismatch(client):
    r = register(client, password="abcdef", confirmPassword="different")
    assert r.status_code == 400


def test_register_short_password(client):
    r = register(client, password="abc", confirmPassword="abc")
    assert r.status_code == 400


def test_register_missing_name(client):
    r = client.post("/api/register", json={
        "firstName": "", "lastName": "", "email": "x@example.com",
        "password": "TestPass123", "confirmPassword": "TestPass123",
    })
    assert r.status_code == 400


def test_login_success(client):
    register(client)
    client.post("/api/logout")
    r = login(client)
    assert r.status_code == 200
    assert r.get_json()["user"]["email"] == "user@example.com"


def test_login_wrong_password(client):
    register(client)
    client.post("/api/logout")
    r = login(client, password="WrongPass1")
    assert r.status_code == 401


def test_login_nonexistent_user(client):
    r = login(client, email="nobody@example.com")
    assert r.status_code == 401


def test_logout_requires_login(client):
    r = client.post("/api/logout")
    assert r.status_code == 401


def test_logout_then_me(client):
    register(client)
    r = client.post("/api/logout")
    assert r.status_code == 200
    r2 = client.get("/api/me")
    assert r2.get_json()["authenticated"] is False


def test_profile_requires_login(client):
    r = client.get("/api/profile")
    assert r.status_code == 401


def test_profile_get(client):
    register(client)
    r = client.get("/api/profile")
    assert r.status_code == 200
    assert r.get_json()["user"]["firstname"] == "Test"


def test_profile_update(client):
    register(client)
    r = client.put("/api/profile", json={
        "firstName": "Updated", "lastName": "Name", "email": "user@example.com",
        "dateOfBirth": "1990-01-01", "sex": "male",
    })
    assert r.status_code == 200
    assert r.get_json()["user"]["firstname"] == "Updated"


def test_profile_update_email_conflict(client):
    register(client, email="a@example.com")
    client.post("/api/logout")
    register(client, email="b@example.com")
    r = client.put("/api/profile", json={
        "firstName": "B", "lastName": "User", "email": "a@example.com",
        "dateOfBirth": "1990-01-01", "sex": "male",
    })
    assert r.status_code == 400