import pytest

import Autism


@pytest.fixture
def app(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(Autism, "DB_PATH", str(db_path))

    flask_app = Autism.create_app()
    flask_app.config["TESTING"] = True

    yield flask_app

    with flask_app.app_context():
        Autism.db.session.remove()
        Autism.db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, email="user@example.com", password="TestPass123", **overrides):
    payload = {
        "firstName": "Test",
        "lastName": "User",
        "email": email,
        "dateOfBirth": "1990-01-01",
        "sex": "male",
        "password": password,
        "confirmPassword": password,
    }
    payload.update(overrides)
    return client.post("/api/register", json=payload)


def login(client, email="user@example.com", password="TestPass123"):
    return client.post("/api/login", json={"email": email, "password": password})


def login_admin(client):
    return client.post("/api/login", json={"email": "admin1", "password": "s7s1234567A"})