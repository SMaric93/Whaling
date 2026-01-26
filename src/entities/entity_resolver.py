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
from config import CROSSWALKS_DIR
from parsing.string_normalizer import normalize_name, normalize_vessel_name

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
    
    def resolve_voyages_df(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        return result
    
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
