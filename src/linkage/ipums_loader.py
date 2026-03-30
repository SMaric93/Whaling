"""
IPUMS data loader for census microdata.

Loads and prepares IPUMS USA Full Count / MLP extracts for captain linkage.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_IPUMS, STAGING_DIR, WHALING_STATE_FIPS, LINKAGE_CONFIG
from parsing.string_normalizer import normalize_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPUMSLoader:
    """
    Loader for IPUMS USA census microdata.
    
    Supports:
    - Fixed-width (DAT) files with codebook
    - CSV extracts
    - MLP linked data with HIK identifier
    """
    
    # Expected IPUMS columns (flexible naming)
    COLUMN_MAP = {
        "year": ["YEAR", "year"],
        "serial": ["SERIAL", "serial"],
        "pernum": ["PERNUM", "pernum"],
        "hik": ["HIK", "hik", "HISTID"],
        "histid": ["HISTID", "histid"],
        "name": ["NAMEFRST", "NAMELAST", "NAME", "name"],
        "namefrst": ["NAMEFRST", "namefrst"],
        "namelast": ["NAMELAST", "namelast"],
        "age": ["AGE", "age"],
        "sex": ["SEX", "sex"],
        "bpl": ["BPL", "bpl", "BIRTHPLACE"],
        "statefip": ["STATEFIP", "statefip", "STATEICP"],
        "county": ["COUNTY", "county", "COUNTYICP"],
        "occ": ["OCC", "occ", "OCC1950", "OCCUPATION"],
        "realprop": ["REALPROP", "realprop", "REALVALUE"],
        "persprop": ["PERSPROP", "persprop", "PERSVALUE"],
        "sploc": ["SPLOC", "sploc"],
        "momloc": ["MOMLOC", "momloc"],
        "poploc": ["POPLOC", "poploc"],
        "relate": ["RELATE", "relate"],
    }

    FILE_PATTERNS = (
        "*.csv",
        "*.csv.gz",
        "*.CSV",
        "*.CSV.GZ",
        "*.dat",
        "*.DAT",
        "*.dta",
        "*.DTA",
        "*.parquet",
        "*.PARQUET",
        "*.zip",
        "*.ZIP",
    )
    
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or RAW_IPUMS
        self._raw_df: Optional[pd.DataFrame] = None
        self._parsed_df: Optional[pd.DataFrame] = None
    
    def _find_column(self, df: pd.DataFrame, field: str) -> Optional[str]:
        """Find column by trying multiple possible names."""
        candidates = self.COLUMN_MAP.get(field, [field])
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _expected_columns(self) -> List[str]:
        """Return the union of relevant IPUMS columns for selective loading."""
        columns: set[str] = set()
        for candidates in self.COLUMN_MAP.values():
            columns.update(candidates)
        return sorted(columns)

    def _iter_source_files(self) -> List[Path]:
        """Find likely IPUMS extract files recursively under the raw directory."""
        seen: set[Path] = set()
        files: List[Path] = []
        for pattern in self.FILE_PATTERNS:
            for filepath in self.raw_dir.rglob(pattern):
                if filepath in seen:
                    continue
                seen.add(filepath)
                if filepath.is_dir():
                    continue
                lower_name = filepath.name.lower()
                if "cb" in lower_name or "readme" in lower_name or filepath.suffix.lower() == ".xml":
                    continue
                files.append(filepath)
        return sorted(files)

    def _read_delimited_file(self, filepath: Path) -> pd.DataFrame:
        """Read CSV-like IPUMS extracts with column projection and chunking."""
        compression = "infer" if filepath.suffix.lower() in {".zip", ".gz"} or filepath.name.lower().endswith(".csv.gz") else None

        header = pd.read_csv(
            filepath,
            nrows=0,
            low_memory=False,
            compression=compression,
        )
        available_columns = [col for col in header.columns if col in self._expected_columns()]

        reader = pd.read_csv(
            filepath,
            usecols=available_columns or None,
            low_memory=False,
            compression=compression,
            chunksize=200_000,
        )
        return pd.concat(reader, ignore_index=True)

    def _read_source_file(self, filepath: Path) -> pd.DataFrame:
        """Read one supported IPUMS extract file."""
        suffix = filepath.suffix.lower()

        if suffix == ".parquet":
            return pd.read_parquet(filepath)
        if suffix == ".dta":
            return pd.read_stata(filepath)
        if suffix in {".csv", ".dat", ".zip", ".gz"} or filepath.name.lower().endswith(".csv.gz"):
            return self._read_delimited_file(filepath)

        raise ValueError(f"Unsupported IPUMS file type: {filepath.name}")
    
    def _load_files(self) -> pd.DataFrame:
        """Load IPUMS extract files."""
        dfs = []

        for filepath in self._iter_source_files():
            logger.info(f"Loading {filepath.relative_to(self.raw_dir)}")

            try:
                df = self._read_source_file(filepath)
                df["_source_file"] = filepath.name
                dfs.append(df)
                logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"  Could not load {filepath.name}: {e}")
        
        if not dfs:
            logger.warning(f"No IPUMS files found in {self.raw_dir}")
            return pd.DataFrame()
        
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined {len(dfs)} files: {len(combined)} total rows")
        
        return combined
    
    def parse(
        self,
        filter_whaling_states: bool = True,
        target_years: Optional[List[int]] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Parse IPUMS data into standardized format.
        
        Args:
            filter_whaling_states: If True, filter to whaling port states
            target_years: List of census years to include
            force_reload: If True, reload even if already parsed
            
        Returns:
            DataFrame with ipums_person_year schema
        """
        if self._parsed_df is not None and not force_reload:
            return self._parsed_df
        
        raw = self._load_files()
        self._raw_df = raw
        
        if len(raw) == 0:
            logger.warning("No IPUMS data loaded")
            return pd.DataFrame()
        
        parsed = pd.DataFrame()
        
        # Extract standard fields
        year_col = self._find_column(raw, "year")
        if year_col:
            parsed["YEAR"] = pd.to_numeric(raw[year_col], errors="coerce").astype("Int64")
        
        serial_col = self._find_column(raw, "serial")
        if serial_col:
            parsed["SERIAL"] = raw[serial_col]
        
        pernum_col = self._find_column(raw, "pernum")
        if pernum_col:
            parsed["PERNUM"] = pd.to_numeric(raw[pernum_col], errors="coerce").astype("Int64")
        
        # HIK (MLP identifier)
        hik_col = self._find_column(raw, "hik")
        if hik_col:
            parsed["HIK"] = raw[hik_col].astype(str)
            logger.info("HIK column found - MLP data available")
        else:
            # Create synthetic HIK from HISTID or YEAR+SERIAL+PERNUM
            histid_col = self._find_column(raw, "histid")
            if histid_col:
                parsed["HIK"] = raw[histid_col].astype(str)
            else:
                parsed["HIK"] = (
                    parsed["YEAR"].astype(str) + "_" +
                    parsed["SERIAL"].astype(str) + "_" +
                    parsed["PERNUM"].astype(str)
                )
            logger.info("Created synthetic HIK from available identifiers")
        
        # Name fields
        namefrst_col = self._find_column(raw, "namefrst")
        namelast_col = self._find_column(raw, "namelast")
        
        if namefrst_col and namelast_col:
            parsed["NAMEFRST"] = raw[namefrst_col].astype(str)
            parsed["NAMELAST"] = raw[namelast_col].astype(str)
            parsed["NAME"] = parsed["NAMEFRST"] + " " + parsed["NAMELAST"]
        else:
            name_col = self._find_column(raw, "name")
            if name_col:
                parsed["NAME"] = raw[name_col].astype(str)
                parsed["NAMEFRST"] = parsed["NAME"].apply(
                    lambda x: x.split()[0] if pd.notna(x) and len(x.split()) > 0 else ""
                )
                parsed["NAMELAST"] = parsed["NAME"].apply(
                    lambda x: x.split()[-1] if pd.notna(x) and len(x.split()) > 1 else ""
                )
        
        # Normalize names
        parsed["NAME_CLEAN"] = parsed["NAME"].apply(normalize_name)
        
        # Age
        age_col = self._find_column(raw, "age")
        if age_col:
            parsed["AGE"] = pd.to_numeric(raw[age_col], errors="coerce")
        
        # Sex
        sex_col = self._find_column(raw, "sex")
        if sex_col:
            parsed["SEX"] = raw[sex_col]
        
        # Geography
        bpl_col = self._find_column(raw, "bpl")
        if bpl_col:
            parsed["BPL"] = raw[bpl_col]
        
        statefip_col = self._find_column(raw, "statefip")
        if statefip_col:
            parsed["STATEFIP"] = pd.to_numeric(raw[statefip_col], errors="coerce").astype("Int64")
        
        county_col = self._find_column(raw, "county")
        if county_col:
            parsed["COUNTY"] = raw[county_col]
        
        # Occupation
        occ_col = self._find_column(raw, "occ")
        if occ_col:
            parsed["OCC"] = raw[occ_col]
        
        # Wealth fields
        realprop_col = self._find_column(raw, "realprop")
        if realprop_col:
            parsed["REALPROP"] = pd.to_numeric(raw[realprop_col], errors="coerce")
        
        persprop_col = self._find_column(raw, "persprop")
        if persprop_col:
            parsed["PERSPROP"] = pd.to_numeric(raw[persprop_col], errors="coerce")
        
        # Family links
        sploc_col = self._find_column(raw, "sploc")
        if sploc_col:
            parsed["SPLOC"] = pd.to_numeric(raw[sploc_col], errors="coerce").astype("Int64")
        
        # Filter by years if specified
        if target_years is None:
            target_years = LINKAGE_CONFIG.target_years
        
        if "YEAR" in parsed.columns and target_years:
            original_len = len(parsed)
            parsed = parsed[parsed["YEAR"].isin(target_years)]
            logger.info(f"Filtered to years {target_years}: {len(parsed)} of {original_len}")
        
        # Filter by whaling states if requested
        if filter_whaling_states and "STATEFIP" in parsed.columns:
            whaling_fips = list(WHALING_STATE_FIPS.values())
            original_len = len(parsed)
            parsed = parsed[parsed["STATEFIP"].isin(whaling_fips)]
            logger.info(f"Filtered to whaling states: {len(parsed)} of {original_len}")
        
        self._parsed_df = parsed
        logger.info(f"Parsed {len(parsed)} IPUMS person-year records")
        
        return parsed
    
    def create_household_signatures(self) -> pd.DataFrame:
        """
        Create household signature variables for matching validation.
        
        Uses SPLOC to identify spouse names for cross-validation.
        """
        if self._parsed_df is None:
            self.parse()
        
        df = self._parsed_df.copy()
        
        if "SPLOC" not in df.columns or df["SPLOC"].isna().all():
            logger.warning("SPLOC not available, cannot create spouse signatures")
            df["SPOUSE_NAME"] = None
            return df
        
        # Create lookup by household
        household_members = df.groupby(["YEAR", "SERIAL"]).apply(
            lambda g: dict(zip(g["PERNUM"], g["NAME_CLEAN"]))
        ).to_dict()
        
        def get_spouse_name(row):
            sploc = row.get("SPLOC")
            year = row.get("YEAR")
            serial = row.get("SERIAL")
            
            if pd.isna(sploc) or sploc == 0:
                return None
            
            hh_key = (year, serial)
            if hh_key in household_members:
                return household_members[hh_key].get(sploc)
            return None
        
        df["SPOUSE_NAME"] = df.apply(get_spouse_name, axis=1)
        
        spouse_count = df["SPOUSE_NAME"].notna().sum()
        logger.info(f"Created spouse signatures for {spouse_count} records")
        
        return df
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """Save parsed IPUMS data."""
        if self._parsed_df is None:
            self.parse()
        
        if output_path is None:
            output_path = STAGING_DIR / "ipums_person_year.parquet"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._parsed_df.to_parquet(output_path, index=False)
        logger.info(f"Saved ipums_person_year to {output_path}")
        
        return output_path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self._parsed_df is None:
            self.parse()
        
        df = self._parsed_df
        
        if len(df) == 0:
            return {"status": "no_data"}
        
        return {
            "total_person_years": len(df),
            "unique_hik": df["HIK"].nunique() if "HIK" in df.columns else 0,
            "years": sorted(df["YEAR"].dropna().unique().tolist()) if "YEAR" in df.columns else [],
            "states": df["STATEFIP"].dropna().unique().tolist() if "STATEFIP" in df.columns else [],
            "has_wealth_vars": df.get("REALPROP", pd.Series()).notna().any(),
            "has_spouse_loc": df.get("SPLOC", pd.Series()).notna().any(),
            "age_range": (df["AGE"].min(), df["AGE"].max()) if "AGE" in df.columns else (None, None),
        }


if __name__ == "__main__":
    loader = IPUMSLoader()
    
    try:
        df = loader.parse()
        
        print("\n=== IPUMS Data Summary ===")
        summary = loader.get_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"\nNote: {e}")
        print("IPUMS data requires manual download from usa.ipums.org")
        print("See docs/ipums_extract_instructions.md for details.")
