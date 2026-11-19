from datetime import date, datetime
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from main import app, _validate_upload_filename
from src.genai.llm_client import LLMConfigError
from src.data.databricks_jobs import DatabricksJobsConfigError

client = TestClient(app)


def test_health_endpoint_success():
    with patch("main.check_connection") as mock_check:
        mock_check.return_value = None
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["databricks_connected"] is True


def test_health_endpoint_failure():
    with patch("main.check_connection") as mock_check:
        mock_check.side_effect = Exception("Connection timed out")
        response = client.get("/health")
        assert response.status_code == 500
        assert "Databricks connection check failed" in response.json()["detail"]


def test_get_products():
    with patch("main.fetch_products") as mock_fetch:
        mock_fetch.return_value = [
            {"product_id": "P001", "product_name": "Smart Watch Pro", "category": "Electronics"},
            {"product_id": "P002", "product_name": "Packaged Foods", "category": "Grocery"},
        ]
        response = client.get("/products")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["product_id"] == "P001"


def test_get_stores():
    with patch("main.fetch_stores") as mock_fetch:
        mock_fetch.return_value = [
            {"store_id": "S01", "store_name": "Mumbai Flagship", "city": "Mumbai", "region": "West"}
        ]
        response = client.get("/stores")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["store_id"] == "S01"


def test_get_forecast_flat():
    with patch("main.fetch_forecasts") as mock_fetch:
        mock_fetch.return_value = [
            {
                "forecast_id": "P001_S01_20260101",
                "forecast_date": date(2026, 1, 1),
                "product_id": "P001",
                "category": "Electronics",
                "brand": "Voltix",
                "store_id": "S01",
                "store_name": "Mumbai Flagship",
                "city": "Mumbai",
                "forecast_horizon": 1,
                "predicted_units": 105.0,
                "prediction_timestamp": datetime(2026, 1, 1, 0, 0, 0),
                "model_name": "lgbm",
                "model_version": "1",
            }
        ]
        response = client.get("/forecast?product_id=P001&store_id=S01")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["forecast_id"] == "P001_S01_20260101"


def test_get_forecast_trend_merged():
    with patch("main.fetch_history_actuals") as mock_hist, patch("main.fetch_forecast_predictions") as mock_fc:
        mock_hist.return_value = [
            {"date": date(2025, 12, 31), "units_sold": 95.0, "product_id": "P001", "store_id": "S01"}
        ]
        mock_fc.return_value = [
            {"forecast_date": date(2026, 1, 1), "predicted_units": 105.0, "product_id": "P001", "store_id": "S01"}
        ]
        response = client.get("/forecast?product_id=P001&store_id=S01&include_history=true")
        assert response.status_code == 200
        data = response.json()
        assert data["product_id"] == "P001"
        assert len(data["points"]) == 2
        assert data["points"][0]["actual_units"] == 95.0
        assert data["points"][1]["predicted_units"] == 105.0


def test_get_forecast_trend_missing_params():
    response = client.get("/forecast?product_id=P001&include_history=true")
    assert response.status_code == 400
    assert "requires both product_id and store_id" in response.json()["detail"]


def test_get_kpis():
    with patch("src.data.analytics.get_kpi_summary") as mock_kpis:
        mock_kpis.return_value = {
            "top_category": {"days": 90, "category": "Electronics", "total_units": 12000.0},
            "top_product": {"days": 90, "product_id": "P001", "product_name": "Watch", "total_units": 5000.0},
            "predicted_revenue": {"total_predicted_units": 8000.0, "total_predicted_revenue": 800000.0},
            "at_risk_count": 1,
        }
        response = client.get("/kpis")
        assert response.status_code == 200
        data = response.json()
        assert data["top_category"]["category"] == "Electronics"
        assert data["at_risk_count"] == 1


def test_chat_quick_action():
    with patch("main.run_quick_action") as mock_quick:
        mock_quick.return_value = {
            "answer": "Top category is Electronics with 12,000 units.",
            "data": {"category": "Electronics", "total_units": 12000.0},
            "functions_used": ["get_top_category"],
        }
        response = client.post("/chat/quick", json={"action": "top_category", "params": {}})
        assert response.status_code == 200
        data = response.json()
        assert "Electronics" in data["answer"]
        assert data["functions_used"] == ["get_top_category"]


def test_validate_upload_filename():
    assert _validate_upload_filename("sales_transactions_2026_01.csv") == "sales_transactions_2026_01.csv"
    assert _validate_upload_filename("calendar_marketing_external_2026_01.csv") == "calendar_marketing_external_2026_01.csv"

    with pytest.raises(Exception):
        _validate_upload_filename("random_file.csv")

    with pytest.raises(Exception):
        _validate_upload_filename("sales_transactions.txt")


def test_chat_unconfigured_error():
    with patch("main.run_chat") as mock_chat:
        mock_chat.side_effect = LLMConfigError("Missing AZURE_OPENAI_API_KEY")
        response = client.post("/chat", json={"message": "What is the best product?"})
        assert response.status_code == 503
        assert "Chat service is not configured" in response.json()["detail"]


def test_pipeline_trigger_unconfigured():
    with patch("src.data.databricks_jobs.trigger_job") as mock_trig:
        mock_trig.side_effect = DatabricksJobsConfigError("Missing DATABRICKS_JOB_ID")
        response = client.post("/pipeline/trigger")
        assert response.status_code == 503
        assert "Pipeline trigger is not configured" in response.json()["detail"]


def test_get_forecast_trend_all():
    with patch("main.fetch_distinct_products") as mock_dist, patch("main.fetch_forecasts") as mock_fc:
        mock_dist.return_value = [{"product_id": "P001", "category": "Electronics"}]
        mock_fc.return_value = [
            {
                "forecast_id": "P001_S01_20260101",
                "forecast_date": date(2026, 1, 1),
                "product_id": "P001",
                "category": "Electronics",
                "brand": "Voltix",
                "store_id": "S01",
                "store_name": "Mumbai",
                "city": "Mumbai",
                "forecast_horizon": 1,
                "predicted_units": 100.0,
                "prediction_timestamp": datetime(2026, 1, 1, 0, 0, 0),
                "model_name": "lgbm",
                "model_version": "1",
            }
        ]
        response = client.get("/forecast/trend/all")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["product_id"] == "P001"
        assert len(data[0]["records"]) == 1


def test_explain_forecast_success():
    with patch("main.fetch_forecasts") as mock_fc, \
         patch("main.fetch_recent_sales_features") as mock_feat, \
         patch("main.fetch_product_static") as mock_prod, \
         patch("main.generate_explanation") as mock_gen:

        mock_fc.return_value = [
            {"forecast_date": date(2026, 1, 1), "predicted_units": 100.0, "category": "Electronics", "model_version": "1"}
        ]
        mock_feat.return_value = [
            {"date": date(2025, 12, 31), "units_sold": 90, "rolling_mean_28": 95.0, "promotion_flag": 0}
        ]
        mock_prod.return_value = {"category": "Electronics", "lifecycle_status": "Growing"}
        mock_gen.return_value = {
            "explanation": "Sales expected to increase.",
            "key_drivers": ["Demand growth"],
            "risks_or_opportunities": ["Inventory constraints"],
            "recommendation": "Increase safety stock.",
        }

        response = client.post("/forecast/explain", json={"product_id": "P001", "store_id": "S01"})
        assert response.status_code == 200
        data = response.json()
        assert data["explanation"] == "Sales expected to increase."
        assert data["recommendation"] == "Increase safety stock."
        assert data["forecast_summary"]["total_predicted_units"] == 100.0


def test_pipeline_status():
    with patch("src.data.databricks_jobs.get_run_status") as mock_status:
        mock_status.return_value = {
            "life_cycle_state": "TERMINATED",
            "result_state": "SUCCESS",
            "state_message": "Run finished",
        }
        response = client.get("/pipeline/status/12345")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == 12345
        assert data["result_state"] == "SUCCESS"

