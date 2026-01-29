"""
ML Prediction and Residual Analysis with SHAP.

Uses gradient boosting (XGBoost/LightGBM) to predict voyage outcomes,
then analyzes feature importance and residuals.

Key outputs:
- Out-of-sample R² (predictability assessment)
- SHAP values (feature importance and interactions)
- Residual analysis (unexplained success factors)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


def train_prediction_model(
    df: pd.DataFrame,
    target_col: str = "log_q",
    feature_cols: List[str] = None,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Train a gradient boosting model to predict voyage outcomes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis data.
    target_col : str
        Target variable to predict.
    feature_cols : list
        Feature columns. If None, uses defaults.
    model_type : str
        "xgboost" or "lightgbm".
    test_size : float
        Fraction for test set.
    random_state : int
        Random seed.
        
    Returns
    -------
    Dict with model, predictions, and metrics.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    
    print("\n" + "=" * 70)
    print(f"ML PREDICTION MODEL: {model_type.upper()}")
    print("=" * 70)
    
    # Default features
    if feature_cols is None:
        feature_cols = [
            "alpha_hat", "gamma_hat",
            "log_tonnage", "log_duration",
            "experience", "decade",
        ]
    
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    print(f"Features: {available_cols}")
    
    # Prepare data
    df_valid = df.dropna(subset=[target_col] + available_cols).copy()
    
    X = df_valid[available_cols].values
    y = df_valid[target_col].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train size: {len(X_train):,}")
    print(f"Test size: {len(X_test):,}")
    
    # Train model
    if model_type == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\nPerformance:")
    print(f"  Train R²: {r2_train:.4f} | RMSE: {rmse_train:.4f}")
    print(f"  Test R²:  {r2_test:.4f} | RMSE: {rmse_test:.4f}")
    
    # Feature importance (native)
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(available_cols, model.feature_importances_))
        print("\nFeature Importance (native):")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {feat}: {imp:.4f}")
    else:
        importance = None
    
    return {
        "model": model,
        "model_type": model_type,
        "feature_cols": available_cols,
        "target_col": target_col,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "feature_importance": importance,
        "df_valid": df_valid,
    }


def compute_shap_values(
    model_results: Dict,
    max_samples: int = 1000,
    save_outputs: bool = True,
) -> Dict:
    """
    Compute SHAP values for feature importance and interaction analysis.
    
    Parameters
    ----------
    model_results : Dict
        Output from train_prediction_model.
    max_samples : int
        Max samples for SHAP computation (for speed).
    save_outputs : bool
        Whether to save plots and summaries.
        
    Returns
    -------
    Dict with SHAP values and analysis.
    """
    import shap
    
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)
    
    model = model_results["model"]
    X_test = model_results["X_test"]
    feature_cols = model_results["feature_cols"]
    
    # Subsample for speed
    if len(X_test) > max_samples:
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        X_explain = X_test[idx]
    else:
        X_explain = X_test
    
    print(f"Computing SHAP values for {len(X_explain)} samples...")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    
    # Mean absolute SHAP values (global importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = dict(zip(feature_cols, mean_abs_shap))
    
    print("\nSHAP Feature Importance (mean |SHAP|):")
    for feat, imp in sorted(shap_importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    
    # Interaction detection: correlation of SHAP values
    print("\nSHAP value correlations (interaction proxy):")
    shap_corr = np.corrcoef(shap_values.T)
    
    # Focus on theta-psi interaction
    if "alpha_hat" in feature_cols and "gamma_hat" in feature_cols:
        theta_idx = feature_cols.index("alpha_hat")
        psi_idx = feature_cols.index("gamma_hat")
        theta_psi_corr = shap_corr[theta_idx, psi_idx]
        print(f"  Corr(SHAP_θ, SHAP_ψ): {theta_psi_corr:.4f}")
    else:
        theta_psi_corr = None
    
    results = {
        "shap_values": shap_values,
        "shap_importance": shap_importance,
        "shap_correlation": shap_corr,
        "theta_psi_interaction_proxy": theta_psi_corr,
        "feature_cols": feature_cols,
        "X_explain": X_explain,
    }
    
    if save_outputs:
        output_dir = Path("output/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save importance
        imp_df = pd.DataFrame([
            {"feature": k, "mean_abs_shap": v}
            for k, v in shap_importance.items()
        ]).sort_values("mean_abs_shap", ascending=False)
        imp_df.to_csv(output_dir / "shap_importance.csv", index=False)
        
        # Try to save summary plot
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_explain,
                feature_names=feature_cols,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png", dpi=150)
            plt.close()
            print(f"\nSaved SHAP summary plot to {output_dir}/shap_summary.png")
        except Exception as e:
            print(f"\nCould not save SHAP plot: {e}")
        
        print(f"Saved to {output_dir}")
    
    return results


def analyze_residuals(
    model_results: Dict,
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Analyze prediction residuals to find unexplained success factors.
    
    Parameters
    ----------
    model_results : Dict
        Output from train_prediction_model.
    df : pd.DataFrame
        Original dataframe with additional columns for analysis.
    save_outputs : bool
        Whether to save analysis.
        
    Returns
    -------
    Dict with residual analysis.
    """
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)
    
    df_valid = model_results["df_valid"].copy()
    
    # Get full predictions
    model = model_results["model"]
    feature_cols = model_results["feature_cols"]
    X_full = df_valid[feature_cols].values
    y_full = df_valid[model_results["target_col"]].values
    y_pred_full = model.predict(X_full)
    
    # Residuals
    df_valid["y_pred"] = y_pred_full
    df_valid["residual"] = y_full - y_pred_full
    df_valid["abs_residual"] = np.abs(df_valid["residual"])
    
    print(f"Residual statistics:")
    print(f"  Mean: {df_valid['residual'].mean():.4f}")
    print(f"  Std: {df_valid['residual'].std():.4f}")
    print(f"  Skew: {df_valid['residual'].skew():.4f}")
    
    # Who outperforms predictions?
    df_valid["outperformer"] = df_valid["residual"] > df_valid["residual"].quantile(0.9)
    df_valid["underperformer"] = df_valid["residual"] < df_valid["residual"].quantile(0.1)
    
    print(f"\nTop 10% outperformers (N={df_valid['outperformer'].sum()}):")
    print(f"  Mean residual: +{df_valid[df_valid['outperformer']]['residual'].mean():.3f}")
    
    # Correlation of residuals with other variables
    corr_vars = ["alpha_hat", "gamma_hat", "experience", "decade"]
    corr_vars = [c for c in corr_vars if c in df_valid.columns]
    
    print("\nResidual correlations:")
    residual_corrs = {}
    for var in corr_vars:
        corr = df_valid["residual"].corr(df_valid[var])
        residual_corrs[var] = corr
        print(f"  {var}: {corr:.4f}")
    
    # Captain-level analysis
    if "captain_id" in df_valid.columns:
        captain_resid = df_valid.groupby("captain_id")["residual"].agg(["mean", "std", "count"])
        captain_resid = captain_resid[captain_resid["count"] >= 3]
        
        # Captains who consistently outperform
        consistent_outperformers = captain_resid[captain_resid["mean"] > captain_resid["mean"].quantile(0.9)]
        n_consistent = len(consistent_outperformers)
        
        print(f"\nConsistent outperformers (captains with 3+ voyages, top 10% mean residual):")
        print(f"  N: {n_consistent}")
        if n_consistent > 0:
            print(f"  Mean excess return: +{consistent_outperformers['mean'].mean():.3f}")
    
    results = {
        "residual_mean": float(df_valid["residual"].mean()),
        "residual_std": float(df_valid["residual"].std()),
        "residual_correlations": residual_corrs,
        "n_outperformers": int(df_valid["outperformer"].sum()),
        "df_with_residuals": df_valid,
    }
    
    if save_outputs:
        output_dir = Path("output/ml")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save residual summary
        summary = pd.DataFrame([
            {"metric": "residual_mean", "value": results["residual_mean"]},
            {"metric": "residual_std", "value": results["residual_std"]},
        ] + [
            {"metric": f"corr_{k}", "value": v}
            for k, v in residual_corrs.items()
        ])
        summary.to_csv(output_dir / "residual_analysis.csv", index=False)
        
        print(f"\nSaved to {output_dir}")
    
    return results


def run_ml_prediction_analysis(
    df: pd.DataFrame,
    save_outputs: bool = True,
) -> Dict:
    """
    Run complete ML prediction pipeline with SHAP and residual analysis.
    """
    print("\n" + "=" * 70)
    print("ML PREDICTION ANALYSIS PIPELINE")
    print("=" * 70)
    
    # 1. Train model
    model_results = train_prediction_model(
        df,
        target_col="log_q",
        model_type="xgboost",
    )
    
    # 2. SHAP analysis
    shap_results = compute_shap_values(
        model_results,
        save_outputs=save_outputs,
    )
    
    # 3. Residual analysis
    residual_results = analyze_residuals(
        model_results,
        df,
        save_outputs=save_outputs,
    )
    
    # Create summary markdown
    if save_outputs:
        output_dir = Path("output/ml")
        
        lines = [
            "# ML Prediction Analysis",
            "",
            "## Model Performance",
            "",
            f"| Metric | Train | Test |",
            f"|--------|-------|------|",
            f"| R² | {model_results['r2_train']:.4f} | {model_results['r2_test']:.4f} |",
            f"| RMSE | {model_results['rmse_train']:.4f} | {model_results['rmse_test']:.4f} |",
            "",
            "## SHAP Feature Importance",
            "",
            "| Feature | Mean |SHAP| |",
            "|---------|-------------|",
        ]
        
        for feat, imp in sorted(shap_results["shap_importance"].items(), key=lambda x: -x[1]):
            lines.append(f"| {feat} | {imp:.4f} |")
        
        if shap_results.get("theta_psi_interaction_proxy") is not None:
            lines.extend([
                "",
                f"**θ-ψ Interaction Proxy:** Corr(SHAP_θ, SHAP_ψ) = {shap_results['theta_psi_interaction_proxy']:.4f}",
            ])
        
        lines.extend([
            "",
            "## Residual Analysis",
            "",
            f"- Residual Std: {residual_results['residual_std']:.4f}",
            f"- Top 10% outperformers: {residual_results['n_outperformers']} voyages",
            "",
        ])
        
        with open(output_dir / "ml_prediction_analysis.md", "w") as f:
            f.write("\n".join(lines))
        
        print(f"\nSaved summary to {output_dir}/ml_prediction_analysis.md")
    
    return {
        "model": model_results,
        "shap": shap_results,
        "residuals": residual_results,
    }


if __name__ == "__main__":
    from src.analyses.data_loader import prepare_analysis_sample
    from src.analyses.baseline_production import estimate_r1
    
    df = prepare_analysis_sample()
    r1_results = estimate_r1(df, use_loo_sample=True)
    df = r1_results["df"]
    
    results = run_ml_prediction_analysis(df)
