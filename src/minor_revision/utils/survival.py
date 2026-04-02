"""
Survival model wrappers (thin over lifelines / statsmodels).
"""

import numpy as np
import pandas as pd
from typing import Optional


def fit_aft_weibull(
    durations: np.ndarray,
    covariates: pd.DataFrame,
    event_observed: Optional[np.ndarray] = None,
) -> dict:
    """
    Fit a Weibull AFT model using lifelines.

    Parameters
    ----------
    durations : array — time-to-event
    covariates : DataFrame — predictor columns
    event_observed : array, optional — 1 if event observed, 0 if censored.
                     If None, all events are observed.

    Returns
    -------
    dict with coefficient estimates, SEs, p-values, and the fitted model.
    """
    try:
        from lifelines import WeibullAFTFitter

        df = covariates.copy()
        df["T"] = durations
        if event_observed is not None:
            df["E"] = event_observed
        else:
            df["E"] = 1

        model = WeibullAFTFitter()
        model.fit(df, duration_col="T", event_col="E")

        summary = model.summary
        results = {
            "model": model,
            "summary": summary,
            "params": {},
        }
        for covariate in covariates.columns:
            if covariate in summary.index.get_level_values(-1):
                row = summary.xs(covariate, level=-1)
                if len(row) > 0:
                    row = row.iloc[0]
                    results["params"][covariate] = {
                        "coef": float(row["coef"]),
                        "se": float(row["se(coef)"]),
                        "p": float(row["p"]),
                    }
        return results

    except ImportError:
        print("  ⚠ lifelines not available; skipping AFT Weibull")
        return {"model": None, "params": {}}
    except Exception as e:
        print(f"  ⚠ AFT Weibull failed: {e}")
        return {"model": None, "params": {}}
