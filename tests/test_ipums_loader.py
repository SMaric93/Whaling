from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ipums_loader_parses_zipped_csv_extract(tmp_path):
    from src.linkage.ipums_loader import IPUMSLoader

    raw_dir = tmp_path / "ipums"
    raw_dir.mkdir(parents=True)
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    csv_path = source_dir / "usa_00001.csv"
    pd.DataFrame(
        [
            {
                "YEAR": 1860,
                "SERIAL": 10,
                "PERNUM": 1,
                "HIK": "hik-1",
                "NAMEFRST": "William",
                "NAMELAST": "Smith",
                "AGE": 42,
                "STATEFIP": 25,
                "OCC": "mariner",
                "IGNORED": "x",
            }
        ]
    ).to_csv(csv_path, index=False)

    zip_path = raw_dir / "usa_00001.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname=csv_path.name)

    parsed = IPUMSLoader(raw_dir).parse(filter_whaling_states=False, target_years=[1860])

    assert len(parsed) == 1
    assert parsed.loc[0, "HIK"] == "hik-1"
    assert parsed.loc[0, "NAME_CLEAN"] == "WILLIAM SMITH"
    assert parsed.loc[0, "YEAR"] == 1860


def test_link_captains_uses_staged_ipums_when_raw_extract_missing(monkeypatch, tmp_path):
    from src.pipeline import stage3_merge

    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    raw_ipums_dir = tmp_path / "raw" / "ipums"
    raw_ipums_dir.parent.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "YEAR": 1860,
                "HIK": "hik-1",
                "NAME_CLEAN": "WILLIAM SMITH",
                "NAMELAST": "SMITH",
                "AGE": 42,
                "SEX": 1,
                "STATEFIP": 25,
                "OCC": "MARINER",
            }
        ]
    ).to_parquet(staging_dir / "ipums_person_year.parquet", index=False)

    class FakeCaptainProfiler:
        def __init__(self, voyages_path):
            self.voyages_path = voyages_path

        def build_profiles(self, force_reload=True):
            return pd.DataFrame({"captain_id": ["C1"]})

        def save(self):
            return staging_dir / "captain_profiles.parquet"

        def get_linkage_candidates(self, target_year):
            return pd.DataFrame(
                {
                    "captain_id": ["C1"],
                    "captain_name_clean": ["WILLIAM SMITH"],
                    "first_name": ["WILLIAM"],
                    "last_name": ["SMITH"],
                    "last_name_soundex": ["S530"],
                    "expected_age_min": [35],
                    "expected_age_max": [50],
                    "modal_port": ["NANTUCKET"],
                }
            )

    class FakeRecordLinker:
        def link_captains_to_census(self, candidates, census_data, target_year):
            assert len(census_data) == 1
            assert census_data.iloc[0]["HIK"] == "hik-1"
            return pd.DataFrame(
                {
                    "captain_id": ["C1"],
                    "HIK": ["hik-1"],
                    "link_year": [target_year],
                    "match_score": [0.95],
                    "match_probability": [0.95],
                    "match_method": ["deterministic"],
                    "name_score": [1.0],
                    "age_score": [0.9],
                    "geo_score": [1.0],
                    "occ_score": [1.0],
                    "spouse_validated": [False],
                    "match_rank": [1],
                }
            )

        def save_linkage(self, linkage_df):
            output_path = staging_dir / "captain_to_ipums.parquet"
            linkage_df.to_parquet(output_path, index=False)
            return output_path

    def fail_parse(self, *args, **kwargs):
        raise AssertionError("raw IPUMS loader should not run when staged data exists")

    monkeypatch.setattr("src.config.STAGING_DIR", staging_dir)
    monkeypatch.setattr("src.config.RAW_IPUMS", raw_ipums_dir)
    monkeypatch.setattr("src.linkage.captain_profiler.CaptainProfiler", FakeCaptainProfiler)
    monkeypatch.setattr("src.linkage.record_linker.RecordLinker", FakeRecordLinker)
    monkeypatch.setattr("src.linkage.ipums_loader.IPUMSLoader.parse", fail_parse)

    assert stage3_merge.link_captains() is True
    assert (staging_dir / "captain_to_ipums.parquet").exists()
