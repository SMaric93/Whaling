"""
Validation gate for VLM-extracted WSL commodity prices.

Checks schema, volume, commodity/unit whitelists, and biophysical
price ranges (19th-century USD-denominated whale-oil market bounds).
Strict mode raises PriceValidationError so the pipeline halts before
bad price data flows into analyses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class PriceValidationError(Exception):
    """Raised when extracted prices fail a fatal validation check."""


REQUIRED_COLUMNS: Set[str] = {
    "wsl_price_id", "page_id", "pdf", "year", "issue_date",
    "commodity", "price_mid", "unit",
}


COMMODITY_WHITELIST: Set[str] = {"sperm_oil", "whale_oil", "whalebone"}
UNIT_WHITELIST: Set[str] = {"per_gallon", "per_lb", "per_barrel"}


# (low, high) USD price ranges per commodity×unit. Sourced from historical
# whale-oil price tables (Starbuck 1878 appendix; Davis/Gallman/Gleiter 1997).
PRICE_RANGES: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("sperm_oil", "per_gallon"): (0.20, 3.00),
    ("sperm_oil", "per_barrel"): (6.00, 100.00),
    ("whale_oil", "per_gallon"): (0.15, 2.00),
    ("whale_oil", "per_barrel"): (4.00, 70.00),
    ("whalebone", "per_lb"):     (0.05, 6.00),
}


@dataclass
class PriceValidationConfig:
    min_prices: int = 50
    max_unknown_commodity_share: float = 0.05
    max_unknown_unit_share: float = 0.05


CONFIG = PriceValidationConfig()


@dataclass
class PriceValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        head = f"{'PASS' if self.ok else 'FAIL'} — {len(self.errors)} errors, {len(self.warnings)} warnings"
        lines = [head]
        for err in self.errors:
            lines.append(f"  ✗ {err}")
        for warn in self.warnings:
            lines.append(f"  ! {warn}")
        return "\n".join(lines)


def _check_volume(df: pd.DataFrame, cfg: PriceValidationConfig,
                  result: PriceValidationResult) -> None:
    n = len(df)
    result.metrics["n_prices"] = n
    if n < cfg.min_prices:
        result.errors.append(f"price count {n:,} below floor {cfg.min_prices:,}")


def _check_commodity(df: pd.DataFrame, cfg: PriceValidationConfig,
                     result: PriceValidationResult) -> None:
    if "commodity" not in df.columns:
        return
    unknown = df["commodity"].isna() | (~df["commodity"].isin(COMMODITY_WHITELIST))
    share = float(unknown.mean())
    result.metrics["unknown_commodity_share"] = round(share, 4)
    if share > cfg.max_unknown_commodity_share:
        result.errors.append(
            f"unknown commodity share {share:.2%} exceeds ceiling {cfg.max_unknown_commodity_share:.2%}"
        )


def _check_unit(df: pd.DataFrame, cfg: PriceValidationConfig,
                result: PriceValidationResult) -> None:
    if "unit" not in df.columns:
        return
    unknown = df["unit"].isna() | (~df["unit"].isin(UNIT_WHITELIST))
    share = float(unknown.mean())
    result.metrics["unknown_unit_share"] = round(share, 4)
    if share > cfg.max_unknown_unit_share:
        result.errors.append(
            f"unknown unit share {share:.2%} exceeds ceiling {cfg.max_unknown_unit_share:.2%}"
        )


def _check_price_ranges(df: pd.DataFrame, result: PriceValidationResult) -> None:
    if not {"commodity", "unit", "price_mid"}.issubset(df.columns):
        return
    mid = pd.to_numeric(df["price_mid"], errors="coerce")
    n_neg = int((mid < 0).sum())
    if n_neg > 0:
        result.errors.append(f"{n_neg} rows have negative price_mid")

    for (commodity, unit), (lo, hi) in PRICE_RANGES.items():
        mask = (df["commodity"] == commodity) & (df["unit"] == unit)
        sub = mid[mask].dropna()
        if sub.empty:
            continue
        n_oor = int(((sub < lo) | (sub > hi)).sum())
        result.metrics[f"{commodity}_{unit}_oor"] = n_oor
        if n_oor > 0:
            result.errors.append(
                f"{commodity} ({unit}): {n_oor} rows outside historical range [{lo}, {hi}]"
            )


def validate_wsl_prices(
    prices: pd.DataFrame,
    config: Optional[PriceValidationConfig] = None,
    strict: bool = True,
) -> PriceValidationResult:
    """Validate a VLM-extracted prices DataFrame."""
    cfg = config or CONFIG
    result = PriceValidationResult(ok=True)

    if prices is None or prices.empty:
        result.errors.append("prices DataFrame is empty")
        result.ok = False
        if strict:
            raise PriceValidationError(result.summary())
        return result

    missing = REQUIRED_COLUMNS - set(prices.columns)
    if missing:
        result.errors.append(f"missing required columns: {sorted(missing)}")
        result.ok = False
        if strict:
            raise PriceValidationError(result.summary())
        return result

    _check_volume(prices, cfg, result)
    _check_commodity(prices, cfg, result)
    _check_unit(prices, cfg, result)
    _check_price_ranges(prices, result)

    result.ok = not result.errors
    logger.info("Price validation: %s", result.summary())

    if strict and not result.ok:
        raise PriceValidationError(result.summary())
    return result
