"""Tests for VLM price extraction validation gate."""

from __future__ import annotations

import pandas as pd
import pytest


def _make_prices(n: int = 100, commodity: str = "sperm_oil",
                 price_low: float = 1.0, price_high: float = 1.2,
                 unit: str = "per_gallon") -> pd.DataFrame:
    return pd.DataFrame({
        "wsl_price_id": [f"wsl_price_{i:08x}" for i in range(n)],
        "page_id": [f"wsl_pg_{i % 20:08x}" for i in range(n)],
        "pdf": ["x.pdf"] * n,
        "year": [1843 + (i % 70) for i in range(n)],
        "page": [1] * n,
        "issue_date": pd.to_datetime([f"{1843 + (i % 70)}-01-15" for i in range(n)]),
        "commodity": [commodity] * n,
        "price_low": [price_low] * n,
        "price_high": [price_high] * n,
        "price_mid": [(price_low + price_high) / 2] * n,
        "unit": [unit] * n,
        "currency": ["USD"] * n,
        "_confidence": [0.9] * n,
    })


def test_empty_prices_dataframe_raises() -> None:
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    with pytest.raises(PriceValidationError):
        validate_wsl_prices(pd.DataFrame(), strict=True)


def test_missing_required_columns_raises() -> None:
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    df = _make_prices().drop(columns=["commodity"])
    with pytest.raises(PriceValidationError):
        validate_wsl_prices(df, strict=True)


def test_unknown_commodity_share_above_ceiling_raises() -> None:
    """≥5% rows with commodity=None or off-whitelist is fatal."""
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    df = _make_prices(n=100)
    df.loc[:10, "commodity"] = None  # 11% unknown
    with pytest.raises(PriceValidationError, match="unknown commodity"):
        validate_wsl_prices(df, strict=True)


def test_unknown_unit_share_above_ceiling_raises() -> None:
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    df = _make_prices(n=100)
    df.loc[:10, "unit"] = None
    with pytest.raises(PriceValidationError, match="unknown unit"):
        validate_wsl_prices(df, strict=True)


def test_sperm_oil_price_outside_historical_range_raises() -> None:
    """Sperm oil $/gal in 19th c. was ~$0.30–$2.50. Values of $50 are fatal."""
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    df = _make_prices(n=100, commodity="sperm_oil", price_low=50.0,
                      price_high=55.0, unit="per_gallon")
    df["price_mid"] = 52.5
    with pytest.raises(PriceValidationError, match="sperm_oil"):
        validate_wsl_prices(df, strict=True)


def test_negative_prices_raise() -> None:
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    df = _make_prices(n=100)
    df.loc[0, "price_mid"] = -0.5
    with pytest.raises(PriceValidationError, match="negative"):
        validate_wsl_prices(df, strict=True)


def test_below_volume_floor_raises() -> None:
    from src.utils.price_validators import (
        PriceValidationError, validate_wsl_prices,
    )
    df = _make_prices(n=10)
    with pytest.raises(PriceValidationError, match="below floor"):
        validate_wsl_prices(df, strict=True)


def test_happy_path_passes_with_metrics() -> None:
    from src.utils.price_validators import validate_wsl_prices

    # Mix of three commodities to avoid single-commodity warnings.
    df = pd.concat([
        _make_prices(n=200, commodity="sperm_oil", price_low=1.0, price_high=1.2,
                     unit="per_gallon"),
        _make_prices(n=200, commodity="whale_oil", price_low=0.4, price_high=0.6,
                     unit="per_gallon"),
        _make_prices(n=200, commodity="whalebone", price_low=0.3, price_high=0.5,
                     unit="per_lb"),
    ], ignore_index=True)

    result = validate_wsl_prices(df, strict=False)
    assert result.ok, result.summary()
    assert result.metrics["n_prices"] == 600
    assert "unknown_commodity_share" in result.metrics
