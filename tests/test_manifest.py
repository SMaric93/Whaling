from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_manifest_add_entry_supports_directories(tmp_path):
    from src.download.manifest import ManifestManager

    manifest_path = tmp_path / "manifest.jsonl"
    source_dir = tmp_path / "downloaded"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (source_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (nested_dir / "beta.txt").write_text("beta", encoding="utf-8")

    manifest = ManifestManager(manifest_path=manifest_path)
    entry = manifest.add_entry(
        source_name="Example_Directory_Source",
        download_url="https://example.com/archive.zip",
        local_path=source_dir,
        license_note="Example terms",
    )

    assert entry.local_path == str(source_dir.absolute())
    assert entry.file_hash_sha256
    assert manifest.verify_hash(source_dir)
