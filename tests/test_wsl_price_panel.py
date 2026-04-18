"""Tests for monthly-resampled commodity price panel."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_quotes(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """rows: list of (issue_date, commodity, price_mid)."""
    data = [
        {"wsl_price_id": f"p{i}", "page_id": f"pg{i}", "pdf": "x.pdf",
         "year": int(d[:4]), "issue_date": d,
         "commodity": c, "price_low": v, "price_high": v, "price_mid": v,
         "unit": "per_gallon", "currency": "USD"}
        for i, (d, c, v) in enumerate(rows)
    ]
    df = pd.DataFrame(data)
    df["issue_date"] = pd.to_datetime(df["issue_date"])
    return df


def test_monthly_panel_one_row_per_month_commodity() -> None:
    """Two quotes for sperm_oil in 1850-01 → one panel row with mean."""
    from src.parsing.wsl_price_panel import build_monthly_panel

    quotes = _make_quotes([
        ("1850-01-05", "sperm_oil", 1.10),
        ("1850-01-20", "sperm_oil", 1.30),
    ])
    panel = build_monthly_panel(quotes)

    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["year_month"] == pd.Period("1850-01", freq="M")
    assert row["commodity"] == "sperm_oil"
    assert row["price_mean"] == pytest.approx(1.20)
    assert row["n_quotes"] == 2


def test_monthly_panel_separates_commodities_and_months() -> None:
    from src.parsing.wsl_price_panel import build_monthly_panel

    quotes = _make_quotes([
        ("1850-01-05", "sperm_oil", 1.10),
        ("1850-01-20", "whale_oil", 0.45),
        ("1850-02-15", "sperm_oil", 1.15),
    ])
    panel = build_monthly_panel(quotes)

    panel = panel.sort_values(["year_month", "commodity"]).reset_index(drop=True)
    assert len(panel) == 3
    assert list(panel["commodity"]) == ["sperm_oil", "whale_oil", "sperm_oil"]
    assert panel.loc[2, "price_mean"] == pytest.approx(1.15)


def test_monthly_panel_emits_nulls_for_missing_months_when_gap_fill_requested() -> None:
    """With gap_fill=True, missing year-month × commodity cells appear as NaN rows."""
    from src.parsing.wsl_price_panel import build_monthly_panel

    quotes = _make_quotes([
        ("1850-01-05", "sperm_oil", 1.10),
        ("1850-03-05", "sperm_oil", 1.30),  # skip Feb
    ])
    panel = build_monthly_panel(quotes, gap_fill=True)

    panel = panel.sort_values("year_month").reset_index(drop=True)
    assert len(panel) == 3  # Jan, Feb, Mar
    assert pd.isna(panel.loc[1, "price_mean"])
    assert panel.loc[1, "n_quotes"] == 0


def test_monthly_panel_empty_input_returns_empty_frame() -> None:
    from src.parsing.wsl_price_panel import build_monthly_panel

    panel = build_monthly_panel(pd.DataFrame())
    assert panel.empty
    assert list(panel.columns) == ["year_month", "commodity", "price_mean",
                                    "price_median", "n_quotes"]


def test_monthly_panel_drops_null_commodity_rows() -> None:
    from src.parsing.wsl_price_panel import build_monthly_panel

    quotes = _make_quotes([
        ("1850-01-05", "sperm_oil", 1.10),
        ("1850-01-20", "sperm_oil", 1.30),
    ])
    quotes.loc[len(quotes)] = {
        "wsl_price_id": "bad", "page_id": "pg", "pdf": "x.pdf",
        "year": 1850, "issue_date": pd.Timestamp("1850-01-25"),
        "commodity": None, "price_low": 1.0, "price_high": 1.0, "price_mid": 1.0,
        "unit": "per_gallon", "currency": "USD",
    }
    panel = build_monthly_panel(quotes)
    assert len(panel) == 1
    assert panel.iloc[0]["n_quotes"] == 2
