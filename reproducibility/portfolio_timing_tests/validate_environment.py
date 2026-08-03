#!/usr/bin/env python3
"""Numerical guard: reproduce the real static and dynamic paths before any null draw."""

from __future__ import annotations

import argparse
from pathlib import Path

import cvxpy as cp
import numpy as np

from timing_core import (
    DROProblem, annualised_sharpe, atomic_json, canonical_digest, cell_deltas,
    load_bundle, run_path, runtime_metadata, sample_masks, sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path("data"))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--return-tolerance", type=float, default=1e-6)
    parser.add_argument("--turnover-tolerance", type=float, default=1e-5)
    parser.add_argument("--cell-tolerance", type=float, default=2e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1 or min(args.return_tolerance, args.turnover_tolerance, args.cell_tolerance) <= 0:
        raise ValueError("threads and tolerances must be positive")
    if "MOSEK" not in cp.installed_solvers():
        raise RuntimeError("MOSEK is not installed or licensed")
    manifest, arrays = load_bundle(args.bundle_dir)
    solver = DROProblem(manifest["window_rows"], manifest["n_assets"], args.threads)
    solved: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    guard_arrays: dict[str, np.ndarray] = {}
    for code in ("L", "S"):
        solved[code] = {}
        for strategy, modulation in (
            ("static", np.ones(manifest["n_months"])),
            ("dynamic", arrays[f"{code}_modulation"]),
        ):
            print(f"validating {code} {strategy} ...", flush=True)
            path = run_path(solver, arrays, code, modulation, manifest)
            solved[code][strategy] = path
            return_error = max(np.max(np.abs(
                path[basis] - arrays[f"{code}_expected_{strategy}_{basis}"]))
                for basis in ("gross", "net"))
            turnover_error = np.max(np.abs(
                path["turnover"] - arrays[f"{code}_expected_{strategy}_turnover"]))
            if return_error > args.return_tolerance:
                raise RuntimeError(f"{code} {strategy}: return path mismatch {return_error:.3e}")
            if turnover_error > args.turnover_tolerance:
                raise RuntimeError(f"{code} {strategy}: turnover path mismatch {turnover_error:.3e}")
            for metric, values in path.items():
                guard_arrays[f"{code}_{strategy}_{metric}"] = values
            print(f"PASS {code} {strategy} | max return error={return_error:.3e} | "
                  f"max turnover error={turnover_error:.3e}", flush=True)
    static_paths = {code: solved[code]["static"] for code in ("L", "S")}
    dynamic_paths = {code: solved[code]["dynamic"] for code in ("L", "S")}
    masks = sample_masks(arrays)
    for basis in ("net", "gross"):
        reconstructed = cell_deltas(static_paths, dynamic_paths, masks, basis)
        for cell in manifest["cells12"]:
            error = abs(reconstructed[cell] - manifest[f"observed_cells_{basis}"][cell])
            if error > args.cell_tolerance:
                raise RuntimeError(f"{basis} {cell}: observed-cell mismatch {error:.3e}")
    guard_array_path = args.bundle_dir / f"runtime_guard_paths_{manifest['experiment_digest'][:12]}.npz"
    temporary = guard_array_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **guard_arrays)
    temporary.replace(guard_array_path)
    runtime = runtime_metadata(args.threads); runtime_digest = canonical_digest(runtime)
    real = {code: {strategy: {
        "sharpe_gross": annualised_sharpe(solved[code][strategy]["gross"]),
        "sharpe_net": annualised_sharpe(solved[code][strategy]["net"]),
        "mean_turnover": float(solved[code][strategy]["turnover"].mean()),
    } for strategy in ("static", "dynamic")} for code in ("L", "S")}
    guard = {
        "schema_version": "timing-tests-runtime-guard-v1.0.0",
        "experiment_digest": manifest["experiment_digest"],
        "runtime": runtime, "runtime_digest": runtime_digest,
        "guard_array_file": guard_array_path.name,
        "guard_array_sha256": sha256_file(guard_array_path),
        "tolerances": {"return": args.return_tolerance, "turnover": args.turnover_tolerance,
                       "observed_cell": args.cell_tolerance},
        "real_paths": real,
    }
    atomic_json(guard, args.bundle_dir / "runtime_guard.json")
    print("\nENVIRONMENT GUARD PASS", flush=True)
    print(f"written {args.bundle_dir / 'runtime_guard.json'}", flush=True)
    print(f"runtime_digest={runtime_digest}", flush=True)


if __name__ == "__main__":
    main()
