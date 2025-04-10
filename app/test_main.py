from fastapi.testclient import TestClient

from config import Settings
from main import app, get_settings

client = TestClient(app)


def get_settings_override():
    return Settings(admin_email="testing_admin@example.com")


app.dependency_overrides[get_settings] = get_settings_override


def test_app():
    response = client.get("/about")
    data = response.json()
    assert data ==   {
        "app_name": "Bridge API.",
        "app_version": 1.0,
        "app_decription": "This is RestAPI application for creating and processing Conversations.",
    }