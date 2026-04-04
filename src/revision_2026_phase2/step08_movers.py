"""
Step 8 — Strengthen within-captain mover design + event studies.

8A: Output mover regressions with 8 specs + up/down switches
8B: Event studies with multiple windows, placebo, and diagnostics
"""
from __future__ import annotations
import logging
import numpy as np, pandas as pd
import statsmodels.api as sm
from .config import CFG, INTERMEDIATES_DIR, TABLES_DIR, FIGURES_DIR
from .helpers import (load_positive_analysis_panel, build_tidy_row, rows_to_tidy,
                      save_table, save_md)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def _build_mover_panel():
    df = load_positive_analysis_panel()
    df = df.sort_values(["captain_id","year_out","date_out"]).copy()
    df["prev_agent"] = df.groupby("captain_id")["agent_id"].shift(1)
    df["switched"] = ((df["agent_id"]!=df["prev_agent"])&df["prev_agent"].notna()).astype(int)
    df["voyage_num"] = df.groupby("captain_id").cumcount()
    cap_agents = df.groupby("captain_id")["agent_id"].nunique()
    movers = cap_agents[cap_agents>=2].index
    df["is_mover"] = df["captain_id"].isin(movers).astype(int)
    # Number of switches
    n_switches = df.groupby("captain_id")["switched"].sum()
    one_switch = n_switches[n_switches==1].index
    df["one_switch_only"] = df["captain_id"].isin(one_switch).astype(int)
    # Merge effects
    for fname, col in [("psi_connected_loo.parquet","psi_hat_loo"),
                        ("psi_broad_resid_loo.parquet","psi_broad_resid")]:
        p = INTERMEDIATES_DIR / fname
        if p.exists():
            tmp = pd.read_parquet(p)
            if col in tmp.columns:
                df = df.merge(tmp[["voyage_id",col]], on="voyage_id", how="left")
    for col in ["psi_hat_loo","psi_broad_resid"]:
        if col in df.columns:
            mu,sig = df[col].mean(), df[col].std()
            df[f"{col}_std"] = (df[col]-mu)/sig if sig>0 else 0
    # Delta psi (change from previous voyage)
    if "psi_hat_loo" in df.columns:
        df["prev_psi"] = df.groupby("captain_id")["psi_hat_loo"].shift(1)
        df["delta_psi"] = df["psi_hat_loo"] - df["prev_psi"]
        df["up_switch"] = ((df["switched"]==1)&(df["delta_psi"]>0)).astype(int)
        df["down_switch"] = ((df["switched"]==1)&(df["delta_psi"]<0)).astype(int)
    return df

# =====================================================================
# 8A: Output mover regressions
# =====================================================================
def run_mover_regressions():
    logger.info("="*60+"\nSTEP 8A: OUTPUT MOVER REGRESSIONS\n"+"="*60)
    df = _build_mover_panel()
    movers = df[df["is_mover"]==1].copy()
    # Drop rows with missing psi or log_q to prevent NaN/inf in design matrix
    psi_col_check = "psi_hat_loo_std" if "psi_hat_loo_std" in movers.columns else None
    if psi_col_check:
        movers = movers.dropna(subset=["log_q", psi_col_check]).copy()
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in movers.columns]
    route_col = "ground_or_route" if "ground_or_route" in movers.columns else None
    logger.info("  Movers: %d voyages, %d captains", len(movers), movers["captain_id"].nunique())

    tidy_rows = []
    psi_col = "psi_hat_loo_std" if "psi_hat_loo_std" in movers.columns else None
    if psi_col is None:
        logger.warning("  No psi_hat_loo found; skipping mover regressions")
        return pd.DataFrame()

    specs = []
    # Build base captain FE
    fe_cap = pd.get_dummies(movers["captain_id"], prefix="c", drop_first=True, dtype=int)
    # Specs
    specs.append(("captain_FE", [psi_col], {"captain": fe_cap}, movers))
    if route_col and "decade" in movers.columns:
        movers["rt_dec"] = movers[route_col].astype(str)+"_"+movers["decade"].astype(str)
        fe_rt = pd.get_dummies(movers["rt_dec"], prefix="r", drop_first=True, dtype=int)
        specs.append(("captain_FE+route_decade_FE", [psi_col], {"captain":fe_cap,"route_decade":fe_rt}, movers))
        specs.append(("captain_FE+route_decade_FE+hw", [psi_col]+ctrl, {"captain":fe_cap,"route_decade":fe_rt}, movers))
    # Balanced switchers only
    balanced = movers[movers["one_switch_only"]==1].copy()
    if len(balanced)>100:
        fe_cap_bal = pd.get_dummies(balanced["captain_id"], prefix="c", drop_first=True, dtype=int)
        specs.append(("one_switch_captains", [psi_col]+ctrl, {"captain":fe_cap_bal}, balanced))
    # Vessel FE
    if "vessel_id" in movers.columns:
        fe_vessel = pd.get_dummies(movers["vessel_id"], prefix="v", drop_first=True, dtype=int)
        specs.append(("vessel_FE+captain_FE", [psi_col]+ctrl, {"captain":fe_cap,"vessel":fe_vessel}, movers))

    for spec_name, rhs_vars, fe_dict, sample in specs:
        try:
            valid_rhs = [r for r in rhs_vars if r in sample.columns]
            parts = [sample[valid_rhs]] + list(fe_dict.values())
            X = pd.concat(parts, axis=1)
            X = sm.add_constant(X, has_constant="add")
            y = sample["log_q"]
            m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":sample["captain_id"]})
            if psi_col in m.params.index:
                tidy_rows.append(build_tidy_row(
                    outcome="log_q", sample_name="movers", spec_name=spec_name, term="psi_hat",
                    estimate=float(m.params[psi_col]), std_error=float(m.bse[psi_col]),
                    n_obs=int(m.nobs), n_captains=int(sample["captain_id"].nunique()),
                    n_agents=int(sample["agent_id"].nunique()),
                    fixed_effects=", ".join(fe_dict.keys()), cluster_scheme="captain_id",
                    effect_object_used="psi_connected_loo"))
        except Exception as e:
            logger.warning("  Mover spec %s failed: %s", spec_name, e)

    save_table(rows_to_tidy(tidy_rows), "table_mover_output_preferred")

    # Up/down switches
    up_down_rows = []
    if "up_switch" in movers.columns:
        # Rebuild captain FE on the cleaned sample
        fe_cap2 = pd.get_dummies(movers["captain_id"], prefix="c", drop_first=True, dtype=int)
        for direction, col in [("up_switch","up_switch"),("down_switch","down_switch")]:
            try:
                rhs_ud = [col] + ctrl
                X_ud = pd.concat([movers[rhs_ud].reset_index(drop=True),
                                  fe_cap2.reset_index(drop=True)], axis=1)
                X_ud = sm.add_constant(X_ud, has_constant="add")
                y_ud = movers["log_q"].reset_index(drop=True)
                m_ud = sm.OLS(y_ud, X_ud).fit(
                    cov_type="cluster", cov_kwds={"groups":movers["captain_id"].reset_index(drop=True)})
                if col in m_ud.params.index:
                    up_down_rows.append(build_tidy_row(
                        outcome="log_q", sample_name="movers", spec_name="captain_FE",
                        term=direction, estimate=float(m_ud.params[col]), std_error=float(m_ud.bse[col]),
                        n_obs=int(m_ud.nobs), n_captains=int(movers["captain_id"].nunique()),
                        fixed_effects="captain", cluster_scheme="captain_id",
                        effect_object_used="psi_connected_loo"))
            except Exception as e:
                logger.warning("  Up/down %s failed: %s", direction, e)
    if up_down_rows:
        save_table(rows_to_tidy(up_down_rows), "table_mover_up_down_switches")
    movers[movers["is_mover"]==1][["voyage_id","captain_id","agent_id"]].to_parquet(
        INTERMEDIATES_DIR/"mover_sample_ids.parquet", index=False)
    logger.info("  Mover regressions: %d coefficient rows", len(tidy_rows))
    return rows_to_tidy(tidy_rows)

# =====================================================================
# 8B: Event studies
# =====================================================================
def run_event_studies():
    logger.info("="*60+"\nSTEP 8B: EVENT STUDIES\n"+"="*60)
    df = _build_mover_panel()
    ctrl = [c for c in ["log_tonnage","log_duration"] if c in df.columns]
    # First switch per captain
    switch_df = df[df["switched"]==1].drop_duplicates("captain_id", keep="first")[
        ["captain_id","voyage_num"]].rename(columns={"voyage_num":"switch_voyage"})
    df = df.merge(switch_df, on="captain_id", how="left")
    df["event_time"] = df["voyage_num"] - df["switch_voyage"]

    all_tidy = []
    all_pretrend = []

    for W in CFG.event_windows:
        window = list(range(-W, W+1))
        df_w = df[df["event_time"].isin(window)].copy()
        counts = df_w.groupby("captain_id")["event_time"].nunique()
        balanced = counts[counts==len(window)].index
        df_bal = df_w[df_w["captain_id"].isin(balanced)].copy()
        logger.info("  Window [-%d,+%d]: %d balanced captains, %d obs", W, W, len(balanced), len(df_bal))
        if len(df_bal)<30: continue

        # Direction-specific sub-samples
        datasets = [("all_switches", df_bal)]
        if "delta_psi" in df_bal.columns:
            up = df_bal[df_bal["captain_id"].isin(
                df_bal[df_bal["up_switch"]==1]["captain_id"].unique())].copy()
            down = df_bal[df_bal["captain_id"].isin(
                df_bal[df_bal["down_switch"]==1]["captain_id"].unique())].copy()
            if len(up)>20: datasets.append(("up_switches", up))
            if len(down)>20: datasets.append(("down_switches", down))

        for ds_name, ds in datasets:
            time_dummies = pd.get_dummies(ds["event_time"].astype(int), prefix="t", dtype=int)
            if "t_-1" in time_dummies.columns:
                time_dummies = time_dummies.drop(columns=["t_-1"])
            fe_cap = pd.get_dummies(ds["captain_id"], prefix="c", drop_first=True, dtype=int)
            X = pd.concat([time_dummies, fe_cap], axis=1)
            X = sm.add_constant(X, has_constant="add")
            y = ds["log_q"]
            try:
                m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups":ds["captain_id"]})
                for t in window:
                    if t==-1: continue
                    col_name = f"t_{t}"
                    if col_name in m.params.index:
                        all_tidy.append(build_tidy_row(
                            outcome="log_q", sample_name=f"event_{ds_name}_w{W}",
                            spec_name="captain_FE", term=f"t={t}",
                            estimate=float(m.params[col_name]), std_error=float(m.bse[col_name]),
                            n_obs=len(y), n_captains=len(ds["captain_id"].unique()),
                            fixed_effects="captain", cluster_scheme="captain_id",
                            notes=f"window=[-{W},+{W}], ref=t-1"))
                # Pre-trend joint test
                pre_cols = [f"t_{t}" for t in window if t<-1 and f"t_{t}" in m.params.index]
                if pre_cols:
                    r_matrix = np.zeros((len(pre_cols), len(m.params)))
                    for i, pc in enumerate(pre_cols):
                        r_matrix[i, list(m.params.index).index(pc)] = 1
                    f_test = m.f_test(r_matrix)
                    fval = f_test.fvalue
                    fval = float(fval[0][0]) if hasattr(fval,'__getitem__') and hasattr(fval[0],'__getitem__') else float(fval)
                    pval = float(f_test.pvalue)
                    all_pretrend.append(dict(sample=ds_name, window=W,
                        f_stat=fval, p_value=pval,
                        n_pre_terms=len(pre_cols)))
                # Average post effect
                post_cols = [f"t_{t}" for t in window if t>0 and f"t_{t}" in m.params.index]
                if post_cols:
                    avg_post = np.mean([m.params[c] for c in post_cols])
                    all_tidy.append(build_tidy_row(
                        outcome="log_q", sample_name=f"event_{ds_name}_w{W}",
                        spec_name="avg_post", term="avg_post_effect",
                        estimate=float(avg_post), std_error=0,
                        n_obs=len(y), n_captains=len(ds["captain_id"].unique()),
                        notes=f"average of {len(post_cols)} post-period coefficients"))
            except Exception as e:
                logger.warning("  Event study %s w%d failed: %s", ds_name, W, e)

    # Placebo event study (shift event dates by -3)
    logger.info("  Running placebo event study...")
    if "switch_voyage" in df.columns:
        df_placebo = df.copy()
        df_placebo["event_time"] = df_placebo["voyage_num"] - (df_placebo["switch_voyage"] - 3)
        W_placebo = 2
        window_p = list(range(-W_placebo, W_placebo+1))
        df_pw = df_placebo[df_placebo["event_time"].isin(window_p)].copy()
        counts_p = df_pw.groupby("captain_id")["event_time"].nunique()
        bal_p = counts_p[counts_p==len(window_p)].index
        df_bp = df_pw[df_pw["captain_id"].isin(bal_p)].copy()
        if len(df_bp) > 30:
            td_p = pd.get_dummies(df_bp["event_time"].astype(int), prefix="t", dtype=int)
            if "t_-1" in td_p.columns: td_p = td_p.drop(columns=["t_-1"])
            fe_p = pd.get_dummies(df_bp["captain_id"], prefix="c", drop_first=True, dtype=int)
            X_p = pd.concat([td_p, fe_p], axis=1)
            X_p = sm.add_constant(X_p, has_constant="add")
            try:
                m_p = sm.OLS(df_bp["log_q"], X_p).fit(
                    cov_type="cluster", cov_kwds={"groups":df_bp["captain_id"]})
                for t in window_p:
                    if t==-1: continue
                    cn = f"t_{t}"
                    if cn in m_p.params.index:
                        all_tidy.append(build_tidy_row(
                            outcome="log_q", sample_name="event_placebo_shifted",
                            spec_name="captain_FE", term=f"t={t}",
                            estimate=float(m_p.params[cn]), std_error=float(m_p.bse[cn]),
                            n_obs=len(df_bp), n_captains=len(bal_p),
                            notes="PLACEBO: event dates shifted by -3"))
            except Exception as e:
                logger.warning("  Placebo event study failed: %s", e)

    result = rows_to_tidy(all_tidy)
    save_table(result, "table_event_study_output")
    save_table(pd.DataFrame(all_pretrend) if all_pretrend else pd.DataFrame(), "table_event_study_pretrends")
    # Placebo table
    placebo_df = result[result["sample_name"].str.contains("placebo")]
    if len(placebo_df)>0:
        save_table(placebo_df, "table_event_study_placebo")

    # Figure (main window=2)
    _plot_event_study(result)
    logger.info("  Event studies: %d coefficient rows", len(result))
    return result

def _plot_event_study(result):
    try:
        main = result[(result["sample_name"].str.contains("all_switches_w2"))&
                       (result["spec_name"]=="captain_FE")&
                       result["term"].str.startswith("t=")].copy()
        if main.empty: return
        main["t"] = main["term"].str.replace("t=","").astype(int)
        main = main.sort_values("t")
        # Add reference period
        ref = pd.DataFrame([dict(t=-1, estimate=0, ci_low=0, ci_high=0)])
        main = pd.concat([main, ref]).sort_values("t")
        fig, ax = plt.subplots(figsize=(7,5))
        ax.fill_between(main["t"], main["ci_low"], main["ci_high"], alpha=0.2, color="#3498db")
        ax.plot(main["t"], main["estimate"], "o-", color="#2c3e50", linewidth=2, markersize=8)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.axvline(-0.5, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Event Time (voyage relative to switch)")
        ax.set_ylabel("log(output) relative to t=-1")
        ax.set_title("Event Study: Output Around Agent Switches")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR/"figure_event_study_output.png", dpi=CFG.figure_dpi)
        plt.close(fig)
        main[["t","estimate","std_error","ci_low","ci_high"]].to_csv(
            FIGURES_DIR/"figure_event_study_output.csv", index=False)
    except Exception as e:
        logger.warning("  Event study figure failed: %s", e)

def run_all_mover_tests():
    r1 = run_mover_regressions()
    r2 = run_event_studies()
    return {"movers": r1, "events": r2}
