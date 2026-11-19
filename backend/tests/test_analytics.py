from unittest.mock import patch
from src.data import analytics


def test_pct_change():
    assert analytics._pct_change(120, 100) == 20.0
    assert analytics._pct_change(80, 100) == -20.0
    assert analytics._pct_change(100, 0) is None
    assert analytics._pct_change(None, 100) is None
    assert analytics._pct_change(100, None) is None


@patch("src.data.db.fetch_category_totals")
def test_get_top_category(mock_fetch):
    mock_fetch.return_value = [
        {"category": "Electronics", "total_units": 15000.5},
        {"category": "Grocery", "total_units": 8000.2},
    ]
    result = analytics.get_top_category(days=90)
    assert result["category"] == "Electronics"
    assert result["total_units"] == 15000.5

    mock_fetch.return_value = []
    empty_result = analytics.get_top_category(days=90)
    assert empty_result["category"] is None
    assert empty_result["total_units"] is None


@patch("src.data.db.fetch_product_totals")
def test_get_top_product(mock_fetch):
    mock_fetch.return_value = [
        {"product_id": "P001", "product_name": "Voltix Watch", "total_units": 5200.75}
    ]
    result = analytics.get_top_product(days=90)
    assert result["product_id"] == "P001"
    assert result["product_name"] == "Voltix Watch"
    assert result["total_units"] == 5200.75


@patch("src.data.db.fetch_category_comparison")
def test_compare_categories(mock_fetch):
    mock_fetch.return_value = [
        {"category": "Electronics", "total_units": 10000.0, "avg_daily_units": 111.11},
        {"category": "Clothing", "total_units": 8000.0, "avg_daily_units": 88.88},
    ]
    result = analytics.compare_categories("Electronics", "Clothing")
    assert result["leader"] == "Electronics"
    assert result["category_a"]["name"] == "Electronics"
    assert result["category_b"]["name"] == "Clothing"


@patch("src.data.db.fetch_store_performance")
def test_get_store_performance(mock_fetch):
    mock_fetch.return_value = {
        "store_name": "Mumbai Flagship 01",
        "recent_avg": 250.0,
        "baseline_avg": 200.0,
    }
    result = analytics.get_store_performance("S01")
    assert result["store_name"] == "Mumbai Flagship 01"
    assert result["recent_avg"] == 250.0
    assert result["baseline_avg"] == 200.0
    assert result["pct_change"] == 25.0


@patch("src.data.db.fetch_predicted_revenue")
@patch("src.data.db.fetch_trending_down")
@patch("src.data.db.fetch_product_totals")
@patch("src.data.db.fetch_category_totals")
@patch("src.data.db.connection_scope")
def test_get_kpi_summary(mock_conn, mock_cat, mock_prod, mock_trend, mock_rev):
    mock_conn.return_value.__enter__.return_value = None
    mock_cat.return_value = [{"category": "Electronics", "total_units": 15000.0}]
    mock_prod.return_value = [{"product_id": "P001", "product_name": "Voltix Watch", "total_units": 5000.0}]
    mock_trend.return_value = [
        {"product_id": "P003", "product_name": "Product 3", "avg_mom_growth_pct": -12.5},
        {"product_id": "P004", "product_name": "Product 4", "avg_mom_growth_pct": -15.0},
    ]
    mock_rev.return_value = {"total_predicted_units": 20000.0, "total_predicted_revenue": 1500000.0}

    summary = analytics.get_kpi_summary()
    assert summary["top_category"]["category"] == "Electronics"
    assert summary["top_product"]["product_name"] == "Voltix Watch"
    assert summary["at_risk_count"] == 4
    assert summary["predicted_revenue"]["total_predicted_revenue"] == 1500000.0
