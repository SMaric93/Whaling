#!/usr/bin/env python3
"""
WSL JSONL Merge Script
======================

Merges per-year JSONL files from V2 pipeline into a single unified file.
Also runs post-processing on the merged output.

Usage:
    python scripts/wsl_merge.py /path/to/extracted/ -o /path/to/merged.jsonl

    # With post-processing:
    python scripts/wsl_merge.py /path/to/extracted/ -o merged.jsonl --postprocess
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter


def merge_jsonl(input_dir, output_path, postprocess=False, verbose=False):
    """Merge per-year JSONL files into a single file."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Find all year-specific JSONL files
    year_files = sorted(input_dir.glob("wsl_events_????.jsonl"))

    if not year_files:
        # Try the unified file
        unified = input_dir / "wsl_events.jsonl"
        if unified.exists():
            year_files = [unified]
        else:
            print(f"ERROR: No JSONL files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Found {len(year_files)} JSONL files to merge")

    # Optional: import post-processor
    pp_func = None
    if postprocess:
        sys.path.insert(0, str(Path(__file__).parent))
        from wsl_postprocess import post_process_page
        pp_func = post_process_page
        print("Post-processing enabled")

    stats = Counter()
    total_events = 0
    total_pages = 0

    with open(output_path, "w") as fout:
        for fpath in year_files:
            year_events = 0
            year_pages = 0
            with open(fpath) as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["json_errors"] += 1
                        continue

                    if pp_func:
                        record = pp_func(record)

                    fout.write(json.dumps(record) + "\n")
                    year_events += record.get("n_events", 0)
                    year_pages += 1

                    # Count page types
                    pt = record.get("page_type", "unknown")
                    stats[f"page_type:{pt}"] += 1

            total_events += year_events
            total_pages += year_pages
            if verbose:
                print(f"  {fpath.name}: {year_pages:,} pages, {year_events:,} events")

    stats["total_pages"] = total_pages
    stats["total_events"] = total_events

    print(f"\nMerge complete:")
    print(f"  Files merged:    {len(year_files)}")
    print(f"  Total pages:     {total_pages:,}")
    print(f"  Total events:    {total_events:,}")
    print(f"  Output:          {output_path}")
    if stats.get("json_errors"):
        print(f"  JSON errors:     {stats['json_errors']:,}")

    print(f"\nPage type distribution:")
    for key in sorted(stats):
        if key.startswith("page_type:"):
            print(f"  {key[10:]}: {stats[key]:,}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge WSL per-year JSONL files")
    parser.add_argument("input_dir", help="Directory containing per-year JSONL files")
    parser.add_argument("-o", "--output", required=True, help="Output merged JSONL file")
    parser.add_argument("--postprocess", action="store_true",
                        help="Run post-processing on merged output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    merge_jsonl(args.input_dir, args.output,
                postprocess=args.postprocess, verbose=args.verbose)


if __name__ == "__main__":
    main()
