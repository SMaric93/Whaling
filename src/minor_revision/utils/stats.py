"""
Statistical helpers: significance stars, formatting, p-values.
"""

import numpy as np
from scipy import stats


def stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    return ""


def fmt_coef(val: float, se: float, p: float) -> str:
    """Format a coefficient with stars and parenthesised SE."""
    return f"{val:.4f}{stars(p)} ({se:.4f})"


def ols_results(y: np.ndarray, X: np.ndarray) -> dict:
    """
    Run OLS via numpy and return results dict.

    Parameters
    ----------
    y : array of shape (n,)
    X : array of shape (n, k) — should include constant if desired.

    Returns
    -------
    dict with keys: beta, se, t, p, r2, n, k, residuals, sigma2
    """
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    sigma2 = float(np.sum(resid ** 2) / (n - k))
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    t_stat = beta / np.where(se > 0, se, 1.0)
    p_val = 2.0 * (1 - stats.t.cdf(np.abs(t_stat), df=max(n - k, 1)))
    r2 = float(1.0 - np.var(resid) / np.var(y)) if np.var(y) > 0 else 0.0

    return {
        "beta": beta,
        "se": se,
        "t": t_stat,
        "p": p_val,
        "r2": r2,
        "n": n,
        "k": k,
        "residuals": resid,
        "sigma2": sigma2,
    }
