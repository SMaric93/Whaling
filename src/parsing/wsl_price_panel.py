"""Monthly-resampled commodity price panel from VLM price quotes."""

from __future__ import annotations

import pandas as pd

PANEL_COLUMNS = ["year_month", "commodity", "price_mean", "price_median", "n_quotes"]


def build_monthly_panel(
    prices: pd.DataFrame,
    gap_fill: bool = False,
) -> pd.DataFrame:
    """
    Collapse per-quote price observations into a (year_month × commodity) panel.

    Args:
        prices: DataFrame with at least ``issue_date``, ``commodity``, ``price_mid``.
        gap_fill: If True, emit NaN rows for month×commodity cells missing
                  between the first and last observed month. Useful for
                  downstream time-series joins.

    Returns:
        DataFrame with columns [year_month (Period[M]), commodity,
        price_mean, price_median, n_quotes].
    """
    if prices is None or prices.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df = prices.dropna(subset=["commodity", "issue_date"]).copy()
    if df.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.dropna(subset=["issue_date"])
    df["year_month"] = df["issue_date"].dt.to_period("M")

    grouped = df.groupby(["year_month", "commodity"])["price_mid"]
    panel = grouped.agg(
        price_mean="mean", price_median="median", n_quotes="count"
    ).reset_index()

    if gap_fill and not panel.empty:
        months = pd.period_range(panel["year_month"].min(),
                                  panel["year_month"].max(), freq="M")
        commodities = panel["commodity"].unique()
        full_index = pd.MultiIndex.from_product(
            [months, commodities], names=["year_month", "commodity"]
        )
        panel = (panel.set_index(["year_month", "commodity"])
                       .reindex(full_index)
                       .reset_index())
        panel["n_quotes"] = panel["n_quotes"].fillna(0).astype(int)

    return panel[PANEL_COLUMNS]
