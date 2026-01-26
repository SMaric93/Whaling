"""
Captain-year level analysis file assembly.

Aggregates voyage outcomes to captain-year panel and merges with IPUMS census data.

Produces:
- analysis_captain_year.parquet with whaling outcomes + census wealth data
- Sensitivity variants (strict/medium/all links)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STAGING_DIR, FINAL_DIR, LINKAGE_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptainAssembler:
    """
    Assembles captain-year wealth panel.
    
    Workflow:
    1. Map voyages → captain_id
    2. Join captain_id → HIK via captain_to_ipums crosswalk
    3. Aggregate voyage outcomes by (captain_id, census_year)
    4. Merge onto ipums_person_year by (HIK, YEAR)
    """
    
    def __init__(
        self,
        staging_dir: Optional[Path] = None,
        final_dir: Optional[Path] = None,
    ):
        self.staging_dir = staging_dir or STAGING_DIR
        self.final_dir = final_dir or FINAL_DIR
        
        self._voyages: Optional[pd.DataFrame] = None
        self._linkage: Optional[pd.DataFrame] = None
        self._ipums: Optional[pd.DataFrame] = None
        self._assembled: Optional[pd.DataFrame] = None
    
    def load_components(self):
        """Load all required components."""
        # Load analysis_voyage (has captain outcomes)
        voyage_path = self.final_dir / "analysis_voyage.parquet"
        if not voyage_path.exists():
            # Try staging
            voyage_path = self.staging_dir / "voyages_master.parquet"
        
        if voyage_path.exists():
            self._voyages = pd.read_parquet(voyage_path)
            logger.info(f"Loaded voyages: {len(self._voyages)} rows")
        else:
            raise FileNotFoundError(f"Voyages not found at {voyage_path}")
        
        # Load captain-to-IPUMS linkage
        linkage_path = self.staging_dir / "captain_to_ipums.parquet"
        if linkage_path.exists():
            self._linkage = pd.read_parquet(linkage_path)
            logger.info(f"Loaded linkage: {len(self._linkage)} rows")
        else:
            logger.warning("Captain-to-IPUMS linkage not found, census merge will be skipped")
            self._linkage = None
        
        # Load IPUMS person-year
        ipums_path = self.staging_dir / "ipums_person_year.parquet"
        if ipums_path.exists():
            self._ipums = pd.read_parquet(ipums_path)
            logger.info(f"Loaded IPUMS: {len(self._ipums)} rows")
        else:
            logger.warning("IPUMS data not found, census merge will be skipped")
            self._ipums = None
    
    def aggregate_voyage_outcomes(
        self,
        window_years: int = 10,
    ) -> pd.DataFrame:
        """
        Aggregate voyage outcomes to captain-year level.
        
        For each captain and census year, aggregates outcomes from voyages
        within a window before the census.
        
        Args:
            window_years: Number of years before census to aggregate
            
        Returns:
            DataFrame with captain-year aggregated outcomes
        """
        if self._voyages is None:
            self.load_components()
        
        df = self._voyages[self._voyages["captain_id"].notna()].copy()
        
        # For each census year, aggregate prior voyages
        aggregations = []
        
        for census_year in LINKAGE_CONFIG.target_years:
            year_start = census_year - window_years
            
            # Voyages ending before census year
            prior_voyages = df[
                (df["year_in"].notna()) &
                (df["year_in"] >= year_start) &
                (df["year_in"] <= census_year)
            ].copy()
            
            if len(prior_voyages) == 0:
                continue
            
            # Aggregate by captain
            agg = prior_voyages.groupby("captain_id").agg(
                whaling_voyages_count=("voyage_id", "count"),
                whaling_total_oil_bbl=("q_oil_bbl", "sum"),
                whaling_total_bone_lbs=("q_bone_lbs", "sum"),
                whaling_mean_oil_per_voyage=("q_oil_bbl", "mean"),
                whaling_mean_duration_days=("duration_days", "mean"),
                whaling_first_voyage_year=("year_out", "min"),
                whaling_last_voyage_year=("year_in", "max"),
            ).reset_index()
            
            # Add VQI if available
            if "vqi_proxy" in prior_voyages.columns:
                vqi_agg = prior_voyages.groupby("captain_id")["vqi_proxy"].mean()
                agg["whaling_mean_vqi_proxy"] = agg["captain_id"].map(vqi_agg)
            
            # Add desertion rate if available
            if "desertion_rate" in prior_voyages.columns:
                deser_agg = prior_voyages.groupby("captain_id")["desertion_rate"].mean()
                agg["whaling_mean_desertion_rate"] = agg["captain_id"].map(deser_agg)
            
            # Add Arctic exposure if available
            if "frac_days_in_arctic_polygon" in prior_voyages.columns:
                arctic_agg = prior_voyages.groupby("captain_id")["frac_days_in_arctic_polygon"].mean()
                agg["whaling_arctic_exposure"] = agg["captain_id"].map(arctic_agg)
            
            agg["census_year"] = census_year
            agg["window_years"] = window_years
            
            # Career duration as of census
            agg["whaling_career_years_as_of"] = census_year - agg["whaling_first_voyage_year"]
            
            aggregations.append(agg)
        
        if not aggregations:
            logger.warning("No voyage aggregations created")
            return pd.DataFrame()
        
        captain_year = pd.concat(aggregations, ignore_index=True)
        logger.info(f"Created {len(captain_year)} captain-year observations")
        
        return captain_year
    
    def assemble(
        self,
        window_years: int = 10,
        linkage_threshold: float = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Assemble captain-year analysis file.
        
        Args:
            window_years: Years before census to aggregate
            linkage_threshold: Minimum match probability to use (default: medium)
            force_reload: If True, reload data
            
        Returns:
            Complete captain-year panel
        """
        if self._assembled is not None and not force_reload:
            return self._assembled
        
        self.load_components()
        
        if linkage_threshold is None:
            linkage_threshold = LINKAGE_CONFIG.medium_threshold
        
        # Aggregate voyage outcomes
        captain_year = self.aggregate_voyage_outcomes(window_years)
        
        if len(captain_year) == 0:
            logger.warning("No captain-year data to assemble")
            return pd.DataFrame()
        
        result = captain_year.copy()
        
        # Join linkage if available
        if self._linkage is not None:
            # Filter linkage by threshold and keep best match per captain
            valid_links = self._linkage[
                (self._linkage["match_probability"] >= linkage_threshold) &
                (self._linkage["match_rank"] == 1)
            ].copy()
            
            logger.info(f"Using {len(valid_links)} captain-census links (threshold {linkage_threshold})")
            
            # Join linkage
            result = result.merge(
                valid_links[["captain_id", "link_year", "HIK", "match_probability", "match_method"]],
                left_on=["captain_id", "census_year"],
                right_on=["captain_id", "link_year"],
                how="left"
            )
            
            linked_count = result["HIK"].notna().sum()
            logger.info(f"Captain-year records with census link: {linked_count} of {len(result)}")
        
        # Join IPUMS data if available
        if self._ipums is not None and "HIK" in result.columns:
            ipums_cols = ["HIK", "YEAR", "NAME_CLEAN", "AGE", "REALPROP", "PERSPROP", 
                          "STATEFIP", "COUNTY", "OCC"]
            ipums_cols = [c for c in ipums_cols if c in self._ipums.columns]
            
            ipums_subset = self._ipums[ipums_cols].drop_duplicates(subset=["HIK", "YEAR"])
            
            result = result.merge(
                ipums_subset,
                left_on=["HIK", "census_year"],
                right_on=["HIK", "YEAR"],
                how="left"
            )
            
            # Compute wealth variables
            if "REALPROP" in result.columns and "PERSPROP" in result.columns:
                result["total_wealth"] = result["REALPROP"].fillna(0) + result["PERSPROP"].fillna(0)
                result["has_wealth_data"] = (
                    result["REALPROP"].notna() | result["PERSPROP"].notna()
                )
            
            wealth_count = result["has_wealth_data"].sum() if "has_wealth_data" in result.columns else 0
            logger.info(f"Captain-year records with wealth data: {wealth_count}")
        
        # Add flags
        result["has_census_link"] = result.get("HIK", pd.Series()).notna()
        result["has_voyage_data"] = True  # All records have this by construction
        
        self._assembled = result
        logger.info(f"Assembled captain-year: {len(result)} rows, {len(result.columns)} columns")
        
        return result
    
    def save(
        self,
        output_path: Optional[Path] = None,
        save_variants: bool = True,
    ) -> Path:
        """
        Save assembled captain-year file.
        
        Args:
            output_path: Optional custom output path
            save_variants: If True, also save sensitivity variants
        """
        if self._assembled is None:
            self.assemble()
        
        if output_path is None:
            output_path = self.final_dir / "analysis_captain_year.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main file
        self._assembled.to_parquet(output_path, index=False)
        logger.info(f"Saved analysis_captain_year to {output_path}")
        
        csv_path = output_path.with_suffix(".csv")
        self._assembled.to_csv(csv_path, index=False)
        
        # Save sensitivity variants if requested
        if save_variants and self._linkage is not None:
            self._save_sensitivity_variants(output_path.parent)
        
        return output_path
    
    def _save_sensitivity_variants(self, output_dir: Path):
        """Save strict/medium/all linkage variants."""
        for threshold, suffix in [
            (LINKAGE_CONFIG.strict_threshold, "_strict_links"),
            (LINKAGE_CONFIG.medium_threshold, "_medium_links"),
            (0.0, "_all_links"),
        ]:
            variant = self.assemble(linkage_threshold=threshold, force_reload=True)
            
            variant_path = output_dir / f"analysis_captain_year{suffix}.parquet"
            variant.to_parquet(variant_path, index=False)
            logger.info(f"Saved {suffix} variant: {len(variant)} rows")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get assembly summary statistics."""
        if self._assembled is None:
            self.assemble()
        
        df = self._assembled
        
        return {
            "total_captain_years": len(df),
            "unique_captains": df["captain_id"].nunique(),
            "census_years": sorted(df["census_year"].unique().tolist()),
            "census_link_rate": df["has_census_link"].mean() if "has_census_link" in df.columns else 0,
            "wealth_data_rate": df["has_wealth_data"].mean() if "has_wealth_data" in df.columns else 0,
            "mean_voyages_per_captain_year": df["whaling_voyages_count"].mean(),
            "total_oil_recorded": df["whaling_total_oil_bbl"].sum(),
        }


if __name__ == "__main__":
    assembler = CaptainAssembler()
    
    try:
        df = assembler.assemble()
        
        print("\n=== Captain-Year Summary ===")
        summary = assembler.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        assembler.save()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run earlier pipeline stages first.")
