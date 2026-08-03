#!/usr/bin/env python3
"""Report shard coverage and the tail of the latest batch log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    manifest = json.loads((args.bundle_dir / "timing_test_manifest.json").read_text())
    digest = manifest["experiment_digest"][:12]
    shards = sorted(args.results_dir.glob(f"timing_tests_*_{digest}.parquet"))
    shards = [p for p in shards if not p.name.startswith("timing_tests_B")]
    completed: set[int] = set()
    print("BATCH OUTPUTS")
    for path in shards:
        frame = pd.read_parquet(path, columns=["draw"])
        draws = sorted(frame["draw"].astype(int).unique()); completed.update(draws)
        bounds = f"[{draws[0]}, {draws[-1]}]" if draws else "empty"
        print(f"  {path.name}: {len(draws):5d} draws {bounds}")
    expected = set(range(manifest["draws"])); missing = sorted(expected - completed)
    print(f"\nTOTAL: {len(completed)}/{manifest['draws']} ({100*len(completed)/manifest['draws']:.1f}%)")
    if missing:
        print(f"first missing draws: {missing[:20]}")
    else:
        print("coverage complete — run merge_results.py")
    logs = sorted(args.logs_dir.glob("batch_*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        latest = logs[-1]; lines = latest.read_text().splitlines()[-10:]
        print(f"\nLATEST LOG\n  {latest}")
        for line in lines:
            print(f"  {line}")


if __name__ == "__main__":
    main()
