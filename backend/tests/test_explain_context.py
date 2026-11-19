from datetime import date, datetime
from src.genai.explain_context import build_forecast_context, _holidays_in_window


def test_holidays_in_window():
    dates = [date(2026, 1, 1), date(2026, 1, 15), date(2026, 1, 26)]
    holidays = _holidays_in_window(dates)
    assert len(holidays) == 2
    assert holidays[0]["name"] == "New Year"
    assert holidays[1]["name"] == "Republic Day"


def test_build_forecast_context():
    forecast_rows = [
        {"forecast_date": date(2026, 1, 1), "predicted_units": 100.0, "category": "Electronics", "model_version": "1"},
        {"forecast_date": date(2026, 1, 2), "predicted_units": 120.0, "category": "Electronics", "model_version": "1"},
    ]
    recent_features_rows = [
        {
            "date": date(2025, 12, 31),
            "units_sold": 90,
            "rolling_mean_28": 100.0,
            "trend_7d": 1.05,
            "trend_28d": 1.02,
            "yoy_growth_pct": 5.0,
            "mom_growth_pct": 2.0,
            "promotion_flag": 1,
            "days_since_last_promotion": 0,
        }
    ]
    product_static = {
        "category": "Electronics",
        "lifecycle_status": "Growing",
        "demand_class": "Medium Volume",
        "price_elasticity_tier": "high",
    }

    ctx = build_forecast_context(
        product_id="P001",
        store_id="S01",
        forecast_rows=forecast_rows,
        recent_features_rows=recent_features_rows,
        product_static=product_static,
    )

    assert ctx["product_id"] == "P001"
    assert ctx["store_id"] == "S01"
    assert ctx["avg_predicted_units"] == 110.0
    assert ctx["total_predicted_units"] == 220.0
    assert ctx["forecast_horizon_days"] == 2
    assert ctx["category"] == "Electronics"
    assert ctx["recent_actuals"]["rolling_mean_28"] == 100.0
    assert ctx["vs_recent_history_pct_change"] == 10.0

