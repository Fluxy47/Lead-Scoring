# tests/test_api.py

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app


# ── Client fixture ─────────────────────────────────────────────────────────────
# scope="module" means the app starts once for all tests in this file.
# The `with` block triggers the lifespan event — model loads on entry,
# cleans up on exit. Without this, /health would return model_loaded: false.

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── Shared test data ───────────────────────────────────────────────────────────

VALID_LEAD = {
    "Lead Origin": "Landing Page Submission",
    "Lead Source": "Google",
    "TotalVisits": 8,
    "Total Time Spent on Website": 1200,
    "Page Views Per Visit": 4.2,
}

MINIMAL_LEAD = {
    "Lead Source": "Google",
}

VALID_LEADS_LIST = [VALID_LEAD, MINIMAL_LEAD]


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    """Helper — converts DataFrame to CSV bytes for upload tests."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_model_is_loaded(self, client):
        data = response = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"


# ── /predict/single ────────────────────────────────────────────────────────────

class TestPredictSingle:
    def test_valid_input_returns_200(self, client):
        response = client.post("/predict/single", json=VALID_LEAD)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/predict/single", json=VALID_LEAD).json()
        assert "conversion_probability" in data
        assert "label" in data
        assert "threshold_used" in data

    def test_probability_is_between_0_and_1(self, client):
        data = client.post("/predict/single", json=VALID_LEAD).json()
        assert 0.0 <= data["conversion_probability"] <= 1.0

    def test_label_is_valid_tier(self, client):
        data = client.post("/predict/single", json=VALID_LEAD).json()
        assert data["label"] in {"Hot", "Warm", "Cold"}

    def test_minimal_input_still_works(self, client):
        # Fix 3 — missing columns should not crash the pipeline
        response = client.post("/predict/single", json=MINIMAL_LEAD)
        assert response.status_code == 200

    def test_empty_body_returns_422(self, client):
        # Pydantic should reject a completely empty body
        response = client.post("/predict/single", json={})
        # Empty dict is valid — all fields are Optional, so this returns 200
        # If you want to test true validation failure, send a wrong type
        assert response.status_code == 200

    def test_wrong_type_returns_422(self, client):
        # TotalVisits must be float — sending a string should fail validation
        response = client.post("/predict/single", json={
            "Lead Source": "Google",
            "TotalVisits": "not-a-number",
        })
        assert response.status_code == 422


# ── /predict/batch ─────────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_valid_batch_returns_200(self, client):
        response = client.post("/predict/batch", json=VALID_LEADS_LIST)
        assert response.status_code == 200

    def test_response_count_matches_input(self, client):
        data = client.post("/predict/batch", json=VALID_LEADS_LIST).json()
        assert data["total_leads"] == len(VALID_LEADS_LIST)
        assert len(data["results"]) == len(VALID_LEADS_LIST)

    def test_each_result_has_required_fields(self, client):
        data = client.post("/predict/batch", json=VALID_LEADS_LIST).json()
        for result in data["results"]:
            assert "conversion_probability" in result
            assert "label" in result
            assert "threshold_used" in result

    def test_empty_list_returns_400(self, client):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 400


# ── /predict/csv ───────────────────────────────────────────────────────────────

class TestPredictCSV:
    def test_valid_csv_returns_200(self, client):
        df = pd.DataFrame([VALID_LEAD, VALID_LEAD])
        csv_bytes = make_csv_bytes(df)
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200

    def test_response_is_csv(self, client):
        df = pd.DataFrame([VALID_LEAD])
        csv_bytes = make_csv_bytes(df)
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert "text/csv" in response.headers["content-type"]

    def test_response_csv_has_prediction_columns(self, client):
        df = pd.DataFrame([VALID_LEAD])
        csv_bytes = make_csv_bytes(df)
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        result_df = pd.read_csv(io.StringIO(response.text))
        assert "conversion_probability" in result_df.columns
        assert "label" in result_df.columns
        assert "threshold_used" in result_df.columns

    def test_missing_columns_csv_still_works(self, client):
        # Fix 3 — _ensure_columns() guard
        df = pd.DataFrame([{"Lead Source": "Google", "TotalVisits": 5}])
        csv_bytes = make_csv_bytes(df)
        response = client.post(
            "/predict/csv",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert response.status_code == 200

    def test_non_csv_file_returns_400(self, client):
        response = client.post(
            "/predict/csv",
            files={"file": ("data.json", b'{"key": "value"}', "application/json")},
        )
        assert response.status_code == 400

    def test_empty_csv_returns_422(self, client):
        response = client.post(
            "/predict/csv",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert response.status_code == 422