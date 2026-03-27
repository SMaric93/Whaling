"""
Entity resolver for creating deterministic IDs for vessels, captains, and agents.

Creates stable identifiers from normalized name strings and disambiguation fields.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Set
from dataclasses import dataclass, field
import hashlib
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CROSSWALKS_DIR, ML_SHIFT_CONFIG
from parsing.string_normalizer import normalize_name, normalize_vessel_name
from ml.record_matching import (
    cluster_match_pairs,
    compute_numeric_distance_features,
    compute_text_pair_features,
    fit_match_probability_model,
    score_match_probability,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EntityVariants:
    """Tracks all observed variants of an entity."""
    entity_id: str
    canonical_name: str
    variants: Set[str] = field(default_factory=set)
    disambiguation_fields: Dict[str, Any] = field(default_factory=dict)
    
    def add_variant(self, variant: str):
        if variant and variant != self.canonical_name:
            self.variants.add(variant)


class EntityResolver:
    """
    Creates deterministic entity IDs for vessels, captains, and agents.
    
    Uses normalized names plus disambiguation fields (like home port)
    to create stable, reproducible identifiers.
    """
    
    def __init__(self):
        self._vessel_registry: Dict[str, EntityVariants] = {}
        self._captain_registry: Dict[str, EntityVariants] = {}
        self._agent_registry: Dict[str, EntityVariants] = {}
        self._ml_duplicate_pairs: Dict[str, pd.DataFrame] = {}
    
    @staticmethod
    def _compute_id(components: List[str], prefix: str = "") -> str:
        """Compute a deterministic ID from components."""
        key = "|".join(str(c) for c in components if c)
        hash_val = hashlib.sha256(key.encode()).hexdigest()[:12]
        return f"{prefix}{hash_val}" if prefix else hash_val
    
    def resolve_vessel(
        self,
        vessel_name: Optional[str],
        home_port: Optional[str] = None,
        rig: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve a vessel to a deterministic ID.
        
        Args:
            vessel_name: Raw or normalized vessel name
            home_port: Home port for disambiguation
            rig: Vessel rig type for disambiguation
            
        Returns:
            Vessel ID string, or None if name is empty
        """
        if not vessel_name:
            return None
        
        name_clean = normalize_vessel_name(vessel_name)
        if not name_clean:
            return None
        
        # Home port helps disambiguate vessels with same name
        port_clean = home_port.upper().strip() if home_port else ""
        
        vessel_id = self._compute_id([name_clean, port_clean], prefix="V")
        
        # Track in registry
        if vessel_id not in self._vessel_registry:
            self._vessel_registry[vessel_id] = EntityVariants(
                entity_id=vessel_id,
                canonical_name=name_clean,
                disambiguation_fields={"home_port": port_clean, "rig": rig}
            )
        
        # Add variant
        if vessel_name:
            self._vessel_registry[vessel_id].add_variant(vessel_name)
        
        return vessel_id
    
    def resolve_captain(
        self,
        captain_name: Optional[str],
        career_start_decade: Optional[int] = None,
        modal_port: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve a captain to a deterministic ID.
        
        Args:
            captain_name: Raw or normalized captain name
            career_start_decade: Decade of first voyage (e.g., 1820)
            modal_port: Most common home port
            
        Returns:
            Captain ID string, or None if name is empty
        """
        if not captain_name:
            return None
        
        name_clean = normalize_name(captain_name)
        if not name_clean:
            return None
        
        # Career decade helps disambiguate captains with same name
        decade_str = str(career_start_decade)[:3] if career_start_decade else ""
        port_clean = modal_port.upper().strip() if modal_port else ""
        
        captain_id = self._compute_id([name_clean, decade_str, port_clean], prefix="C")
        
        if captain_id not in self._captain_registry:
            self._captain_registry[captain_id] = EntityVariants(
                entity_id=captain_id,
                canonical_name=name_clean,
                disambiguation_fields={
                    "career_start_decade": career_start_decade,
                    "modal_port": port_clean
                }
            )
        
        if captain_name:
            self._captain_registry[captain_id].add_variant(captain_name)
        
        return captain_id
    
    def resolve_agent(
        self,
        agent_name: Optional[str],
        port: Optional[str] = None,
    ) -> Optional[str]:
        """
        Resolve an agent to a deterministic ID.
        
        Args:
            agent_name: Raw or normalized agent name
            port: Port where agent operates
            
        Returns:
            Agent ID string, or None if name is empty
        """
        if not agent_name:
            return None
        
        name_clean = normalize_name(agent_name)
        if not name_clean:
            return None
        
        port_clean = port.upper().strip() if port else ""
        
        agent_id = self._compute_id([name_clean, port_clean], prefix="A")
        
        if agent_id not in self._agent_registry:
            self._agent_registry[agent_id] = EntityVariants(
                entity_id=agent_id,
                canonical_name=name_clean,
                disambiguation_fields={"port": port_clean}
            )
        
        if agent_name:
            self._agent_registry[agent_id].add_variant(agent_name)
        
        return agent_id
    
    def resolve_voyages_df(
        self,
        df: pd.DataFrame,
        *,
        ml_refine: bool = True,
    ) -> pd.DataFrame:
        """
        Add entity IDs to a voyages dataframe.
        
        Args:
            df: DataFrame with vessel_name_clean, captain_name_clean, agent_name_clean columns
            
        Returns:
            DataFrame with vessel_id, captain_id, agent_id columns added
        """
        result = df.copy()
        
        # First pass: collect career info for captains
        captain_careers = {}
        if "captain_name_clean" in df.columns and "year_out" in df.columns:
            for _, row in df.iterrows():
                captain = row.get("captain_name_clean")
                year = row.get("year_out")
                port = row.get("home_port") or row.get("port_out")
                
                if pd.notna(captain) and pd.notna(year):
                    if captain not in captain_careers:
                        captain_careers[captain] = {"years": [], "ports": []}
                    captain_careers[captain]["years"].append(int(year))
                    if pd.notna(port):
                        captain_careers[captain]["ports"].append(port)
        
        # Compute career start decades and modal ports
        captain_info = {}
        for captain, info in captain_careers.items():
            years = info["years"]
            ports = info["ports"]
            
            decade = (min(years) // 10) * 10 if years else None
            modal_port = max(set(ports), key=ports.count) if ports else None
            captain_info[captain] = {"decade": decade, "modal_port": modal_port}
        
        # Resolve entities
        def resolve_vessel_row(row):
            return self.resolve_vessel(
                row.get("vessel_name_clean"),
                home_port=row.get("home_port"),
                rig=row.get("rig")
            )
        
        def resolve_captain_row(row):
            captain = row.get("captain_name_clean")
            info = captain_info.get(captain, {})
            return self.resolve_captain(
                captain,
                career_start_decade=info.get("decade"),
                modal_port=info.get("modal_port")
            )
        
        def resolve_agent_row(row):
            return self.resolve_agent(
                row.get("agent_name_clean"),
                port=row.get("home_port") or row.get("port_out")
            )
        
        result["vessel_id"] = result.apply(resolve_vessel_row, axis=1)
        result["captain_id"] = result.apply(resolve_captain_row, axis=1)
        result["agent_id"] = result.apply(resolve_agent_row, axis=1)

        if ml_refine and ML_SHIFT_CONFIG.enabled:
            result = self._apply_ml_entity_refinement(result)
        
        return result

    def _apply_ml_entity_refinement(self, voyages_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse likely duplicate entity ids using a learned pairwise model."""
        refined = voyages_df.copy()
        entity_specs = [
            ("vessel", "vessel_id", "vessel_name_clean", ["home_port", "rig"], "year_out"),
            ("captain", "captain_id", "captain_name_clean", ["home_port", "port_out"], "year_out"),
            ("agent", "agent_id", "agent_name_clean", ["home_port", "port_out"], "year_out"),
        ]

        for entity_type, id_col, name_col, aux_cols, year_col in entity_specs:
            if id_col not in refined.columns or name_col not in refined.columns:
                continue
            entity_df = self._build_entity_frame(refined, id_col, name_col, aux_cols, year_col)
            if len(entity_df) < 2:
                continue

            pair_scores = self._score_duplicate_entity_pairs(entity_df, entity_type)
            self._ml_duplicate_pairs[entity_type] = pair_scores

            if len(pair_scores) == 0:
                continue

            mapping = cluster_match_pairs(
                pair_scores,
                left_col="left_id",
                right_col="right_id",
                probability_col="match_probability",
                threshold=ML_SHIFT_CONFIG.entity_merge_probability_threshold,
            )
            if not mapping:
                continue

            refined[id_col] = refined[id_col].map(
                lambda value: mapping.get(str(value), value) if pd.notna(value) else value
            )

        return refined

    def _build_entity_frame(
        self,
        voyages_df: pd.DataFrame,
        id_col: str,
        name_col: str,
        aux_cols: List[str],
        year_col: str,
    ) -> pd.DataFrame:
        """Aggregate voyage-level entities into one row per current entity id."""
        work = voyages_df[[col for col in [id_col, name_col, year_col] + aux_cols if col in voyages_df.columns]].copy()
        work = work[work[id_col].notna() & work[name_col].notna()].copy()
        if len(work) == 0:
            return pd.DataFrame(columns=[id_col, name_col, "obs_count"])

        def _mode(series: pd.Series):
            clean = series.dropna()
            if len(clean) == 0:
                return None
            return clean.mode().iloc[0]

        agg_map = {name_col: _mode}
        if year_col in work.columns:
            agg_map[year_col] = "min"
        agg_map.update({col: _mode for col in aux_cols if col in work.columns})

        grouped = work.groupby(id_col, dropna=False).agg(agg_map).reset_index()
        counts = work.groupby(id_col).size().rename("obs_count").reset_index()
        grouped = grouped.merge(counts, on=id_col, how="left")
        return grouped

    def _score_duplicate_entity_pairs(
        self,
        entity_df: pd.DataFrame,
        entity_type: str,
    ) -> pd.DataFrame:
        """Score likely duplicate pairs for vessels, captains, or agents."""
        id_col = f"{entity_type}_id"
        name_col = {
            "vessel": "vessel_name_clean",
            "captain": "captain_name_clean",
            "agent": "agent_name_clean",
        }[entity_type]
        port_col = "home_port" if "home_port" in entity_df.columns else ("port_out" if "port_out" in entity_df.columns else None)
        year_col = "year_out" if "year_out" in entity_df.columns else None
        rig_col = "rig" if "rig" in entity_df.columns else None

        entity_df = entity_df.copy()
        entity_df["block"] = entity_df[name_col].fillna("").astype(str).str[:2]

        pair_rows = []
        for _, block_df in entity_df.groupby("block", dropna=False):
            block_df = block_df.head(ML_SHIFT_CONFIG.max_blocking_candidates)
            rows = block_df.to_dict("records")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    left = rows[i]
                    right = rows[j]
                    features = compute_text_pair_features(left.get(name_col), right.get(name_col), prefix="name_")
                    if port_col:
                        features.update(compute_text_pair_features(left.get(port_col), right.get(port_col), prefix="port_"))
                    if year_col:
                        features.update(compute_numeric_distance_features(left.get(year_col), right.get(year_col), scale=30.0, prefix="year_"))
                    if rig_col:
                        features["rig_match"] = float((left.get(rig_col) or "") == (right.get(rig_col) or "") and pd.notna(left.get(rig_col)) and pd.notna(right.get(rig_col)))
                    features["obs_count_ratio"] = min(left.get("obs_count", 1), right.get("obs_count", 1)) / max(left.get("obs_count", 1), right.get("obs_count", 1), 1)

                    rule_score = (
                        0.65 * features["name_jw"] +
                        0.15 * features.get("port_exact", 0.5) +
                        0.10 * features.get("year_similarity", 0.5) +
                        0.10 * features.get("rig_match", 0.5)
                    )
                    pair_rows.append({
                        "left_id": str(left[id_col]),
                        "right_id": str(right[id_col]),
                        "left_name": left.get(name_col),
                        "right_name": right.get(name_col),
                        "heuristic_score": float(rule_score),
                        **features,
                    })

        if not pair_rows:
            return pd.DataFrame()

        pair_df = pd.DataFrame(pair_rows)
        feature_cols = [
            col for col in pair_df.columns
            if col not in {"left_id", "right_id", "left_name", "right_name", "heuristic_score"}
        ]
        positives = pair_df["heuristic_score"] >= ML_SHIFT_CONFIG.heuristic_positive_threshold
        negatives = pair_df["heuristic_score"] <= ML_SHIFT_CONFIG.heuristic_negative_threshold
        bundle = fit_match_probability_model(
            pair_df[feature_cols],
            positives,
            negatives,
        )
        pair_df["match_probability"] = score_match_probability(
            bundle,
            pair_df[feature_cols],
            fallback_scores=pair_df["heuristic_score"],
        )
        pair_df["model_trained"] = bundle.trained
        return pair_df.sort_values("match_probability", ascending=False).reset_index(drop=True)

    def _build_entity_crosswalk(
        self,
        voyages_df: pd.DataFrame,
        *,
        entity_type: str,
        id_col: str,
        name_col: str,
        extra_cols: List[str],
    ) -> pd.DataFrame:
        """Build a saved crosswalk with ML merge metadata for one entity type."""
        resolved = self.resolve_voyages_df(voyages_df, ml_refine=True)
        keep_cols = [col for col in [id_col, name_col] + extra_cols if col in resolved.columns]
        crosswalk = resolved[keep_cols].dropna(subset=[id_col, name_col]).drop_duplicates().copy()
        counts = (
            resolved[resolved[id_col].notna()]
            .groupby(id_col)
            .size()
            .rename("observation_count")
            .reset_index()
        )
        crosswalk = crosswalk.merge(counts, on=id_col, how="left")

        pair_scores = self._ml_duplicate_pairs.get(entity_type, pd.DataFrame())
        if len(pair_scores) > 0:
            max_prob = {}
            for _, row in pair_scores.iterrows():
                prob = float(row["match_probability"])
                max_prob[row["left_id"]] = max(prob, max_prob.get(row["left_id"], 0.0))
                max_prob[row["right_id"]] = max(prob, max_prob.get(row["right_id"], 0.0))
            crosswalk["ml_duplicate_probability"] = crosswalk[id_col].astype(str).map(max_prob).fillna(0.0)
        else:
            crosswalk["ml_duplicate_probability"] = 0.0

        return crosswalk.sort_values([id_col, name_col]).reset_index(drop=True)

    def resolve_vessels(self, voyages_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve and summarize vessel entities with ML-assisted duplicate collapsing."""
        return self._build_entity_crosswalk(
            voyages_df,
            entity_type="vessel",
            id_col="vessel_id",
            name_col="vessel_name_clean",
            extra_cols=["home_port", "rig"],
        )

    def resolve_captains(self, voyages_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve and summarize captain entities with ML-assisted duplicate collapsing."""
        return self._build_entity_crosswalk(
            voyages_df,
            entity_type="captain",
            id_col="captain_id",
            name_col="captain_name_clean",
            extra_cols=["home_port", "port_out", "year_out"],
        )

    def resolve_agents(self, voyages_df: pd.DataFrame) -> pd.DataFrame:
        """Resolve and summarize agent entities with ML-assisted duplicate collapsing."""
        return self._build_entity_crosswalk(
            voyages_df,
            entity_type="agent",
            id_col="agent_id",
            name_col="agent_name_clean",
            extra_cols=["home_port", "port_out"],
        )
    
    def save_registries(self, output_dir: Optional[Path] = None):
        """Save entity registries to crosswalks directory."""
        output_dir = output_dir or CROSSWALKS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Vessel registry
        vessel_records = []
        for entity_id, entity in self._vessel_registry.items():
            vessel_records.append({
                "vessel_id": entity_id,
                "canonical_name": entity.canonical_name,
                "variants": "|".join(sorted(entity.variants)),
                "home_port": entity.disambiguation_fields.get("home_port"),
                "rig": entity.disambiguation_fields.get("rig"),
            })
        
        if vessel_records:
            pd.DataFrame(vessel_records).to_csv(
                output_dir / "vessel_registry.csv", index=False
            )
            logger.info(f"Saved {len(vessel_records)} vessels to registry")
        
        # Captain registry
        captain_records = []
        for entity_id, entity in self._captain_registry.items():
            captain_records.append({
                "captain_id": entity_id,
                "canonical_name": entity.canonical_name,
                "variants": "|".join(sorted(entity.variants)),
                "career_start_decade": entity.disambiguation_fields.get("career_start_decade"),
                "modal_port": entity.disambiguation_fields.get("modal_port"),
            })
        
        if captain_records:
            pd.DataFrame(captain_records).to_csv(
                output_dir / "captain_registry.csv", index=False
            )
            logger.info(f"Saved {len(captain_records)} captains to registry")
        
        # Agent registry
        agent_records = []
        for entity_id, entity in self._agent_registry.items():
            agent_records.append({
                "agent_id": entity_id,
                "canonical_name": entity.canonical_name,
                "variants": "|".join(sorted(entity.variants)),
                "port": entity.disambiguation_fields.get("port"),
            })
        
        if agent_records:
            pd.DataFrame(agent_records).to_csv(
                output_dir / "agent_registry.csv", index=False
            )
            logger.info(f"Saved {len(agent_records)} agents to registry")
    
    def get_stats(self) -> Dict[str, int]:
        """Get entity resolution statistics."""
        return {
            "unique_vessels": len(self._vessel_registry),
            "unique_captains": len(self._captain_registry),
            "unique_agents": len(self._agent_registry),
            "total_vessel_variants": sum(
                len(e.variants) + 1 for e in self._vessel_registry.values()
            ),
            "total_captain_variants": sum(
                len(e.variants) + 1 for e in self._captain_registry.values()
            ),
        }
