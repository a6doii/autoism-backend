from tests.conftest import register, login, login_admin


def test_questions_public_no_auth_required(client):
    r = client.get("/api/questions")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["questions"]) == 10
    assert "white-european" in body["ethnicities"]


def test_cases_requires_login(client):
    r = client.get("/api/cases")
    assert r.status_code == 401


def test_create_and_list_case(client):
    register(client)
    r = client.post("/api/cases", json={
        "child_name": "Child A", "child_dob": "2020-01-01", "brief": "brief text",
    })
    assert r.status_code == 200
    case_id = r.get_json()["case"]["id"]

    r = client.get("/api/cases")
    assert r.status_code == 200
    cases = r.get_json()["cases"]
    assert any(c["id"] == case_id for c in cases)


def test_create_case_missing_fields(client):
    register(client)
    r = client.post("/api/cases", json={"child_name": "", "child_dob": "", "brief": ""})
    assert r.status_code == 400


def test_case_detail_get_update_delete(client):
    register(client)
    case_id = client.post("/api/cases", json={
        "child_name": "Child A", "child_dob": "2020-01-01", "brief": "b",
    }).get_json()["case"]["id"]

    r = client.get(f"/api/cases/{case_id}")
    assert r.status_code == 200

    r = client.put(f"/api/cases/{case_id}", json={"child_name": "Renamed", "child_dob": "2020-01-01", "brief": "b2"})
    assert r.status_code == 200
    assert r.get_json()["case"]["child_name"] == "Renamed"

    r = client.delete(f"/api/cases/{case_id}")
    assert r.status_code == 200

    r = client.get(f"/api/cases/{case_id}")
    assert r.status_code == 404


def test_case_not_found(client):
    register(client)
    r = client.get("/api/cases/99999")
    assert r.status_code == 404


def test_user_cannot_access_other_users_case(client):
    register(client, email="owner@example.com")
    case_id = client.post("/api/cases", json={
        "child_name": "Owner Child", "child_dob": "2020-01-01", "brief": "b",
    }).get_json()["case"]["id"]
    client.post("/api/logout")

    register(client, email="intruder@example.com")
    r = client.get(f"/api/cases/{case_id}")
    assert r.status_code == 403

    r = client.delete(f"/api/cases/{case_id}")
    assert r.status_code == 403


def test_admin_can_see_all_cases(client):
    register(client, email="owner@example.com")
    client.post("/api/cases", json={"child_name": "Owner Child", "child_dob": "2020-01-01", "brief": "b"})
    client.post("/api/logout")

    login_admin(client)
    r = client.get("/api/cases")
    assert r.status_code == 200
    assert any(c["child_name"] == "Owner Child" for c in r.get_json()["cases"])


def test_assessment_test_endpoint_accepts_empty_payload(client):
    """Documents a real validation gap: POST /cases/<id>/test with no
    answers and no image still returns 200 with a fabricated 0%-risk
    report instead of rejecting the request with 400."""
    register(client)
    case_id = client.post("/api/cases", json={
        "child_name": "Child A", "child_dob": "2020-01-01", "brief": "b",
    }).get_json()["case"]["id"]

    r = client.post(f"/api/cases/{case_id}/test", data={})
    assert r.status_code == 200
    body = r.get_json()["result"]
    assert body["combined_risk"] < 0.01


def test_reports_empty_for_new_case(client):
    register(client)
    case_id = client.post("/api/cases", json={
        "child_name": "Child A", "child_dob": "2020-01-01", "brief": "b",
    }).get_json()["case"]["id"]
    r = client.get(f"/api/cases/{case_id}/reports")
    assert r.status_code == 200
    assert r.get_json()["reports"] == []


def test_report_detail_forbidden_for_other_user(client):
    register(client, email="owner@example.com")
    case_id = client.post("/api/cases", json={
        "child_name": "Child A", "child_dob": "2020-01-01", "brief": "b",
    }).get_json()["case"]["id"]
    client.post(f"/api/cases/{case_id}/test", data={})
    report_id = client.get(f"/api/cases/{case_id}/reports").get_json()["reports"][0]["id"]
    client.post("/api/logout")

    register(client, email="intruder@example.com")
    r = client.get(f"/api/reports/{report_id}")
    assert r.status_code == 403


def test_admin_dashboard_requires_admin(client):
    register(client)
    r = client.get("/api/admin/dashboard")
    assert r.status_code == 403


def test_admin_dashboard_as_admin(client):
    login_admin(client)
    r = client.get("/api/admin/dashboard")
    assert r.status_code == 200
    assert "total_accounts" in r.get_json()["stats"]


def test_admin_users_requires_admin(client):
    register(client)
    r = client.get("/api/admin/users")
    assert r.status_code == 403


def test_admin_can_delete_regular_user(client):
    register(client, email="deleteme@example.com")
    client.post("/api/logout")
    login_admin(client)
    users = client.get("/api/admin/users").get_json()["users"]
    target = next(u for u in users if u["email"] == "deleteme@example.com")
    r = client.delete(f"/api/admin/users/{target['id']}")
    assert r.status_code == 200


def test_admin_account_cannot_be_deleted(client):
    login_admin(client)
    users = client.get("/api/admin/users").get_json()["users"]
    admin_id = next(u for u in users if u["email"] == "admin1")["id"]
    r = client.delete(f"/api/admin/users/{admin_id}")
    assert r.status_code == 400


def test_validate_face_image_requires_login(client):
    r = client.post("/api/validate-face-image", data={})
    assert r.status_code == 401


def test_validate_face_image_no_file(client):
    register(client)
    r = client.post("/api/validate-face-image", data={})
    assert r.status_code == 400
    assert r.get_json()["valid"] is False