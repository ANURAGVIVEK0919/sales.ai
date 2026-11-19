from datetime import date, datetime
import pytest
from pydantic import ValidationError

from src.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ExplainRequest,
    ExplainResponse,
    ForecastListResponse,
    ForecastPoint,
    ForecastRecord,
    ForecastSummary,
    ForecastTrendResponse,
    HealthResponse,
    KPISummary,
    PipelineStatusResponse,
    PipelineTriggerResponse,
    PipelineUploadAndTriggerResponse,
    PredictedRevenueStat,
    Product,
    ProductForecastGroup,
    QuickActionRequest,
    Store,
    TopCategoryStat,
    TopProductStat,
    UploadedFileInfo,
)


def test_forecast_record_valid():
    record = ForecastRecord(
        forecast_id="P001_S01_20260101",
        forecast_date=date(2026, 1, 1),
        product_id="P001",
        category="Electronics",
        brand="Voltix",
        store_id="S01",
        store_name="Mumbai Flagship 01",
        city="Mumbai",
        forecast_horizon=1,
        predicted_units=150.5,
        prediction_timestamp=datetime(2026, 1, 1, 0, 0, 0),
        model_name="sales_forecasting_lgbm",
        model_version="1",
    )
    assert record.product_id == "P001"
    assert record.predicted_units == 150.5


def test_forecast_record_invalid():
    with pytest.raises(ValidationError):
        ForecastRecord(
            forecast_id="P001_S01_20260101",
            forecast_date="invalid-date",
            product_id="P001",
        )


def test_health_response():
    resp = HealthResponse(status="ok", databricks_connected=True)
    assert resp.databricks_connected is True


def test_product_and_store_models():
    prod = Product(product_id="P001", product_name="Smart Watch", category="Electronics")
    assert prod.product_id == "P001"

    store = Store(store_id="S01", store_name="Mumbai Store", city="Mumbai", region="West")
    assert store.region == "West"


def test_kpi_summary_model():
    kpi = KPISummary(
        top_category=TopCategoryStat(days=90, category="Electronics", total_units=12000.0),
        top_product=TopProductStat(days=90, product_id="P001", product_name="Smart Watch", total_units=5000.0),
        predicted_revenue=PredictedRevenueStat(total_predicted_units=8000.0, total_predicted_revenue=500000.0),
        at_risk_count=3,
    )
    assert kpi.at_risk_count == 3
    assert kpi.top_category.category == "Electronics"

