"""
Voyage-level analysis file assembly.

Merges:
- voyages_master
- voyage_labor_metrics (crew/desertion)
- voyage_routes (Arctic exposure)
- vessel_register_year (quality proxy, as-of merge)

Outputs analysis_voyage.parquet/.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, FINAL_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoyageAssembler:
    """
    Assembles the final voyage-level analysis file.
    
    Implements the merge logic:
    - voyages_master LEFT JOIN voyage_labor_metrics ON voyage_id
    - LEFT JOIN voyage_routes ON voyage_id
    - ASOF MERGE vessel_register_year ON vessel_id (nearest prior year)
    """
    
    def __init__(
        self,
        staging_dir: Optional[Path] = None,
        final_dir: Optional[Path] = None,
    ):
        self.staging_dir = staging_dir or STAGING_DIR
        self.final_dir = final_dir or FINAL_DIR
        
        self._voyages: Optional[pd.DataFrame] = None
        self._labor_metrics: Optional[pd.DataFrame] = None
        self._routes: Optional[pd.DataFrame] = None
        self._vessel_register: Optional[pd.DataFrame] = None
        self._assembled: Optional[pd.DataFrame] = None
    
    def load_components(self):
        """Load all component tables from staging."""
        # Load voyages master
        voyages_path = self.staging_dir / "voyages_master.parquet"
        if voyages_path.exists():
            self._voyages = pd.read_parquet(voyages_path)
            logger.info(f"Loaded voyages_master: {len(self._voyages)} rows")
        else:
            raise FileNotFoundError(f"voyages_master not found at {voyages_path}")
        
        # Load labor metrics (optional)
        labor_path = self.staging_dir / "voyage_labor_metrics.parquet"
        if labor_path.exists():
            self._labor_metrics = pd.read_parquet(labor_path)
            logger.info(f"Loaded voyage_labor_metrics: {len(self._labor_metrics)} rows")
        else:
            logger.warning("voyage_labor_metrics not found, will skip")
        
        # Load routes (optional)
        routes_path = self.staging_dir / "voyage_routes.parquet"
        if routes_path.exists():
            self._routes = pd.read_parquet(routes_path)
            logger.info(f"Loaded voyage_routes: {len(self._routes)} rows")
        else:
            logger.warning("voyage_routes not found, will skip")
        
        # Load vessel register (optional)
        register_path = self.staging_dir / "vessel_register_year.parquet"
        if register_path.exists():
            self._vessel_register = pd.read_parquet(register_path)
            logger.info(f"Loaded vessel_register_year: {len(self._vessel_register)} rows")
        else:
            logger.warning("vessel_register_year not found, will skip")
    
    def _asof_merge_vessel_register(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform as-of merge to attach vessel quality proxy using pd.merge_asof.
        
        For each voyage, find the vessel register entry with the
        closest year <= year_out (or nearest available if none prior).
        """
        if self._vessel_register is None or len(self._vessel_register) == 0:
            logger.info("No vessel register data, skipping VQI merge")
            df["vqi_proxy"] = np.nan
            df["vqi_asof_year"] = np.nan
            df["vqi_asof_exact"] = False
            df["vqi_asof_extrapolated"] = True
            return df
        
        vessel_reg = self._vessel_register.dropna(subset=["vessel_name_clean", "year", "vqi_proxy"])
        if len(vessel_reg) == 0:
            logger.warning("No usable vessel register entries with VQI proxy")
            df["vqi_proxy"] = np.nan
            df["vqi_asof_year"] = np.nan
            df["vqi_asof_exact"] = False
            df["vqi_asof_extrapolated"] = False
            return df
            
        result = df.copy()
        
        # To use merge_asof, both dataframes must be sorted by the key
        # Handle NaN year_out separately
        nan_years = result[result["year_out"].isna()].copy()
        valid_years = result[result["year_out"].notna()].copy()
        
        valid_years["year_out_sort"] = valid_years["year_out"].astype(int)
        valid_years = valid_years.sort_values("year_out_sort")
        
        vessel_reg_sorted = vessel_reg.sort_values("year").copy()
        vessel_reg_sorted["year"] = vessel_reg_sorted["year"].astype(int)
        
        # Backward match (nearest prior year)
        backward_merged = pd.merge_asof(
            valid_years,
            vessel_reg_sorted[["vessel_name_clean", "year", "vqi_proxy"]],
            left_on="year_out_sort",
            right_on="year",
            by="vessel_name_clean",
            direction="backward",
            suffixes=("", "_bwd")
        )
        
        # Forward match (nearest future year, for those with no prior year)
        forward_merged = pd.merge_asof(
            valid_years,
            vessel_reg_sorted[["vessel_name_clean", "year", "vqi_proxy"]],
            left_on="year_out_sort",
            right_on="year",
            by="vessel_name_clean",
            direction="forward",
            suffixes=("", "_fwd")
        )
        
        # Combine logic: use backward if available, otherwise forward (extrapolated)
        vqi_proxy = backward_merged["vqi_proxy"].fillna(forward_merged["vqi_proxy"])
        asof_year = backward_merged["year"].fillna(forward_merged["year"])
        
        is_exact = (asof_year == valid_years["year_out_sort"].values)
        is_extrap = backward_merged["vqi_proxy"].isna() & forward_merged["vqi_proxy"].notna()
        
        valid_years["vqi_proxy"] = vqi_proxy.values
        valid_years["vqi_asof_year"] = asof_year.values
        valid_years["vqi_asof_exact"] = is_exact
        valid_years["vqi_asof_extrapolated"] = is_extrap.values
        
        valid_years = valid_years.drop(columns=["year_out_sort"])
        
        # Handle nan_years
        nan_years["vqi_proxy"] = np.nan
        nan_years["vqi_asof_year"] = np.nan
        nan_years["vqi_asof_exact"] = False
        nan_years["vqi_asof_extrapolated"] = True
        
        # Recombine and restore original index order
        result_final = pd.concat([valid_years, nan_years]).loc[result.index]
        
        matched = result_final["vqi_proxy"].notna().sum()
        logger.info(f"VQI as-of merge matched {matched} of {len(result_final)} voyages")
        
        return result_final
    
    def assemble(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Assemble the final analysis_voyage table.
        
        Returns:
            Complete voyage-level analysis DataFrame
        """
        if self._assembled is not None and not force_reload:
            return self._assembled
        
        # Load components
        self.load_components()
        
        # Start with voyages master
        result = self._voyages.copy()
        logger.info(f"Starting with {len(result)} voyages")
        
        # Left join labor metrics
        if self._labor_metrics is not None:
            labor_cols = [c for c in self._labor_metrics.columns if c != "voyage_id"]
            result = result.merge(
                self._labor_metrics[["voyage_id"] + labor_cols],
                on="voyage_id",
                how="left"
            )
            matched = result["crew_count"].notna().sum() if "crew_count" in result.columns else 0
            logger.info(f"Labor metrics merge: {matched} of {len(result)} matched")
        
        # Left join routes
        if self._routes is not None:
            route_cols = [c for c in self._routes.columns if c != "voyage_id"]
            result = result.merge(
                self._routes[["voyage_id"] + route_cols],
                on="voyage_id",
                how="left"
            )
            matched = result["days_observed"].notna().sum() if "days_observed" in result.columns else 0
            logger.info(f"Route metrics merge: {matched} of {len(result)} matched")
        
        # As-of merge vessel register
        result = self._asof_merge_vessel_register(result)
        
        # Compute derived fields
        result = self._compute_derived_fields(result)
        
        self._assembled = result
        logger.info(f"Assembled analysis_voyage: {len(result)} rows, {len(result.columns)} columns")
        
        return result
    
    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived analysis fields."""
        result = df.copy()
        
        # q_total_index: weighted combination of oil and bone
        # Using rough historical price ratios (oil ~$1/gallon, bone ~$1-5/lb variable)
        if "q_oil_bbl" in result.columns and "q_bone_lbs" in result.columns:
            # Normalize to comparable scale
            oil_component = result["q_oil_bbl"].fillna(0)
            bone_component = result["q_bone_lbs"].fillna(0) / 100  # Scale down
            
            result["q_total_index"] = oil_component + bone_component
        
        # Route year cell (already computed in parser, ensure present)
        if "route_year_cell" not in result.columns:
            result["route_year_cell"] = result.apply(
                lambda r: f"{r['ground_or_route']}_{r['year_out']}" 
                if pd.notna(r.get("ground_or_route")) and pd.notna(r.get("year_out")) 
                else None,
                axis=1
            )
        
        # Merge coverage flags
        result["has_labor_data"] = (
            result["crew_count"].notna() if "crew_count" in result.columns else False
        )
        result["has_route_data"] = (
            result["days_observed"].notna() if "days_observed" in result.columns else False
        )
        result["has_vqi_data"] = result["vqi_proxy"].notna()
        
        # Ensure canonical ID columns exist for downstream ML and Stage 4 analysis loaders
        for name_col, id_col in [
            ("captain_name_clean", "captain_id"),
            ("agent_name_clean", "agent_id"),
            ("vessel_name_clean", "vessel_id"),
        ]:
            if id_col not in result.columns and name_col in result.columns:
                result[id_col] = result[name_col]
        
        return result
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save assembled analysis file."""
        if self._assembled is None:
            self.assemble()
        
        if output_path is None:
            output_path = self.final_dir / "analysis_voyage.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._assembled.to_parquet(output_path, index=False)
        logger.info(f"Saved analysis_voyage to {output_path}")
        
        # Also save CSV
        csv_path = output_path.with_suffix(".csv")
        self._assembled.to_csv(csv_path, index=False)
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get assembly summary statistics."""
        if self._assembled is None:
            self.assemble()
        
        df = self._assembled
        
        return {
            "total_voyages": len(df),
            "year_range": (df["year_out"].min(), df["year_out"].max()),
            "labor_data_coverage": df["has_labor_data"].mean() if "has_labor_data" in df.columns else 0,
            "route_data_coverage": df["has_route_data"].mean() if "has_route_data" in df.columns else 0,
            "vqi_data_coverage": df["has_vqi_data"].mean() if "has_vqi_data" in df.columns else 0,
            "total_columns": len(df.columns),
            "unique_captains": df["captain_id"].nunique() if "captain_id" in df.columns else 0,
            "unique_vessels": df["vessel_id"].nunique() if "vessel_id" in df.columns else 0,
        }


if __name__ == "__main__":
    assembler = VoyageAssembler()
    
    try:
        df = assembler.assemble()
        
        print("\n=== Analysis Voyage Summary ===")
        summary = assembler.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        assembler.save()
        
        print("\n=== Column List ===")
        for col in sorted(df.columns):
            print(f"  {col}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the parsing stages first to create staging files.")
