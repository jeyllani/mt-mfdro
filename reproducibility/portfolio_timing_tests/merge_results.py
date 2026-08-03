#!/usr/bin/env python3
"""Merge shards and derive the three inferential levels from validated draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from timing_core import (
    METHOD_BLOCKS, atomic_json, atomic_parquet, cell_deltas, load_bundle,
    sample_masks, sha256_file, validate_draws,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def corrected_p(null: pd.Series | np.ndarray, observed: float) -> tuple[int, float]:
    values = np.asarray(null, dtype=float); exceedances = int(np.count_nonzero(values >= observed))
    return exceedances, float((exceedances + 1) / (len(values) + 1))


def main() -> None:
    args = parse_args(); manifest, arrays = load_bundle(args.bundle_dir)
    digest_short = manifest["experiment_digest"][:12]
    shards = sorted(args.results_dir.glob(f"timing_tests_*_{digest_short}.parquet"))
    shards = [p for p in shards if not p.name.startswith("timing_tests_B")]
    if not shards:
        raise FileNotFoundError("no timing-test shards found")
    frames = [pd.read_parquet(path).assign(_source=path.name) for path in shards]
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined[combined.duplicated("draw", keep=False)]
    for draw, group in duplicates.groupby("draw"):
        comparison = group.drop(columns="_source").reset_index(drop=True)
        if not comparison.eq(comparison.iloc[0]).all().all():
            raise RuntimeError(f"conflicting duplicate draw {draw}")
    combined = combined.drop_duplicates("draw", keep="first").drop(columns="_source")
    expected = set(range(manifest["draws"]))
    combined = validate_draws(combined, manifest, expected, require_complete=True)
    if combined["runtime_digest"].nunique() != 1:
        raise RuntimeError("shards were produced by different numerical runtimes")

    guard = json.loads((args.bundle_dir / "runtime_guard.json").read_text())
    if guard["experiment_digest"] != manifest["experiment_digest"]:
        raise RuntimeError("runtime guard experiment mismatch")
    guard_array_path = args.bundle_dir / guard["guard_array_file"]
    if sha256_file(guard_array_path) != guard["guard_array_sha256"]:
        raise RuntimeError("runtime guard array SHA-256 mismatch")
    with np.load(guard_array_path, allow_pickle=False) as loaded:
        guard_paths = {name: loaded[name] for name in loaded.files}
    static_paths = {code: {metric: guard_paths[f"{code}_static_{metric}"]
                           for metric in ("gross", "net", "turnover")} for code in ("L", "S")}
    dynamic_paths = {code: {metric: guard_paths[f"{code}_dynamic_{metric}"]
                            for metric in ("gross", "net", "turnover")} for code in ("L", "S")}
    masks = sample_masks(arrays)
    observed = {basis: cell_deltas(static_paths, dynamic_paths, masks, basis)
                for basis in ("net", "gross")}
    target = manifest["target_cell"]

    # Level 1: three separate timing-placebo distributions for the real large-cap dynamic Sharpe.
    placebo_rows = []
    for method in METHOD_BLOCKS:
        for basis in ("net", "gross"):
            real = float(guard["real_paths"]["L"]["dynamic"][f"sharpe_{basis}"])
            column = f"L_{method}_sharpe_{basis}"
            r, p = corrected_p(combined[column], real)
            placebo_rows.append({
                "level": "placebo", "method": method, "block_months": METHOD_BLOCKS[method],
                "basis": basis, "observed_sharpe": real, "exceedances": r,
                "draws": len(combined), "p_value": p,
                "null_mean": float(combined[column].mean()),
                "null_q95": float(combined[column].quantile(.95)),
                "experiment_digest": manifest["experiment_digest"],
            })
    placebo = pd.DataFrame(placebo_rows)

    # Levels 2 and 3 plus the four marginal tests, all from the SAME monthly joint draws.
    cell_rows = []
    for basis in ("net", "gross"):
        for cell in manifest["cells12"]:
            r_marg, p_marg = corrected_p(combined[f"{basis}_{cell}"], observed[basis][cell])
            r_12, p_12 = corrected_p(combined[f"{basis}_maxT12"], observed[basis][cell])
            if cell in manifest["cells4"]:
                r_4, p_4 = corrected_p(combined[f"{basis}_maxT4"], observed[basis][cell])
            else:
                r_4, p_4 = None, np.nan
            cell_rows.append({
                "cell": cell, "basis": basis, "observed_delta_sharpe": observed[basis][cell],
                "marginal_exceedances": r_marg, "p_marginal": p_marg,
                "maxT4_exceedances": r_4, "p_maxT4": p_4,
                "maxT12_exceedances": r_12, "p_maxT12": p_12,
                "is_target": cell == target, "draws": len(combined),
                "experiment_digest": manifest["experiment_digest"],
            })
    inference = pd.DataFrame(cell_rows)
    target_net = inference[(inference["basis"] == manifest["primary_basis"]) & inference["is_target"]]
    if len(target_net) != 1:
        raise RuntimeError("target cell is not unique")
    target_row = target_net.iloc[0]
    if target_row["p_maxT12"] + 1e-15 < target_row["p_maxT4"]:
        raise RuntimeError("nested-family monotonicity failure: p_maxT12 < p_maxT4")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    draws_out = args.results_dir / f"timing_tests_B{len(combined)}_{digest_short}.parquet"
    placebo_out = args.results_dir / f"level1_placebo_B{len(combined)}_{digest_short}.parquet"
    inference_out = args.results_dir / f"levels2_3_inference_B{len(combined)}_{digest_short}.parquet"
    atomic_parquet(combined, draws_out); atomic_parquet(placebo, placebo_out)
    atomic_parquet(inference, inference_out)
    summary = {
        "schema_version": "portfolio-timing-tests-summary-v1.0.0",
        "experiment_digest": manifest["experiment_digest"], "runtime_digest": guard["runtime_digest"],
        "draws": len(combined), "target_cell": target, "primary_basis": manifest["primary_basis"],
        "level1_placebo": placebo[placebo["basis"] == manifest["primary_basis"]]
            .set_index("method")[["observed_sharpe", "exceedances", "p_value", "null_mean", "null_q95"]]
            .to_dict(orient="index"),
        "level2_maxT4": {
            "target_observed_delta_sharpe": float(target_row["observed_delta_sharpe"]),
            "exceedances": int(target_row["maxT4_exceedances"]),
            "p_value": float(target_row["p_maxT4"]),
        },
        "level3_maxT12": {
            "target_observed_delta_sharpe": float(target_row["observed_delta_sharpe"]),
            "exceedances": int(target_row["maxT12_exceedances"]),
            "p_value": float(target_row["p_maxT12"]),
        },
        "files": {"draws": draws_out.name, "placebo": placebo_out.name, "inference": inference_out.name},
    }
    atomic_json(summary, args.results_dir / "timing_tests_summary.json")

    print("PRIMARY NET INFERENCE")
    print("Level 1 — timing placebo")
    for row in placebo[placebo["basis"] == "net"].itertuples():
        print(f"  {row.method:<7} observed={row.observed_sharpe:+.6f} r={row.exceedances:4d} p={row.p_value:.4f}")
    print("Level 2 — four-cell family")
    primary4 = inference[(inference["basis"] == "net") & inference["cell"].isin(manifest["cells4"])]
    for row in primary4.itertuples():
        print(f"  {row.cell:<7} observed={row.observed_delta_sharpe:+.6f} "
              f"marginal={row.p_marginal:.4f} maxT4={row.p_maxT4:.4f}")
    print(f"  TARGET {target}: p_maxT4={target_row['p_maxT4']:.4f}")
    print("Level 3 — extended 12-cell family")
    print(f"  TARGET {target}: p_maxT12={target_row['p_maxT12']:.4f}")
    print("\nwritten")
    for path in (draws_out, placebo_out, inference_out, args.results_dir / "timing_tests_summary.json"):
        print(f"  {path}")


if __name__ == "__main__":
    main()
