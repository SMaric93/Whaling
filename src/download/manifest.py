"""
Manifest manager for tracking downloaded files with provenance metadata.

Records:
- source_name
- retrieval_date_utc
- download_url
- local_path
- file_hash_sha256
- license_or_terms_note
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MANIFEST_FILE, MANIFEST_FIELDS, LICENSE_NOTES


@dataclass
class ManifestEntry:
    """A single manifest entry for a downloaded file."""
    source_name: str
    retrieval_date_utc: str
    download_url: str
    local_path: str
    file_hash_sha256: str
    license_or_terms_note: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ManifestEntry":
        return cls(**{k: d[k] for k in MANIFEST_FIELDS})


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class ManifestManager:
    """
    Manages the manifest.jsonl file for tracking downloaded data provenance.
    
    Each line in the manifest is a JSON object with the fields specified
    in MANIFEST_FIELDS.
    """
    
    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or MANIFEST_FILE
        self._entries: List[ManifestEntry] = []
        self._load()
    
    def _load(self) -> None:
        """Load existing manifest entries from file."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._entries.append(ManifestEntry.from_dict(data))
    
    def _save(self) -> None:
        """Save all manifest entries to file."""
        with open(self.manifest_path, "w") as f:
            for entry in self._entries:
                f.write(json.dumps(entry.to_dict()) + "\n")
    
    def add_entry(
        self,
        source_name: str,
        download_url: str,
        local_path: Path,
        license_key: Optional[str] = None,
        license_note: Optional[str] = None,
    ) -> ManifestEntry:
        """
        Add a new manifest entry for a downloaded file.
        
        Args:
            source_name: Identifier for the data source (e.g., "AOWV_Voyages")
            download_url: URL the file was downloaded from
            local_path: Local filesystem path where file is stored
            license_key: Key into LICENSE_NOTES dict, or None
            license_note: Custom license note (overrides license_key)
        
        Returns:
            The created ManifestEntry
        """
        # Compute hash
        file_hash = compute_file_hash(local_path)
        
        # Get license note
        if license_note is None:
            license_note = LICENSE_NOTES.get(license_key, "Unknown license")
        
        # Create entry
        entry = ManifestEntry(
            source_name=source_name,
            retrieval_date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            download_url=download_url,
            local_path=str(local_path.absolute()),
            file_hash_sha256=file_hash,
            license_or_terms_note=license_note,
        )
        
        # Check for duplicate (same source and URL)
        self._entries = [
            e for e in self._entries 
            if not (e.source_name == source_name and e.download_url == download_url)
        ]
        
        self._entries.append(entry)
        self._save()
        
        return entry
    
    def get_entries(self, source_name: Optional[str] = None) -> List[ManifestEntry]:
        """
        Get manifest entries, optionally filtered by source name.
        
        Args:
            source_name: Filter to entries with this source name
            
        Returns:
            List of matching ManifestEntry objects
        """
        if source_name is None:
            return self._entries.copy()
        return [e for e in self._entries if e.source_name == source_name]
    
    def verify_hash(self, local_path: Path) -> bool:
        """
        Verify that a file's current hash matches its manifest entry.
        
        Args:
            local_path: Path to the file to verify
            
        Returns:
            True if hash matches, False if not found or mismatch
        """
        path_str = str(local_path.absolute())
        for entry in self._entries:
            if entry.local_path == path_str:
                current_hash = compute_file_hash(local_path)
                return current_hash == entry.file_hash_sha256
        return False
    
    def verify_all(self) -> Dict[str, bool]:
        """
        Verify all files in the manifest.
        
        Returns:
            Dict mapping local_path to verification result (True/False)
        """
        results = {}
        for entry in self._entries:
            path = Path(entry.local_path)
            if path.exists():
                current_hash = compute_file_hash(path)
                results[entry.local_path] = (current_hash == entry.file_hash_sha256)
            else:
                results[entry.local_path] = False
        return results
    
    def __len__(self) -> int:
        return len(self._entries)
    
    def __repr__(self) -> str:
        return f"ManifestManager({len(self._entries)} entries)"
