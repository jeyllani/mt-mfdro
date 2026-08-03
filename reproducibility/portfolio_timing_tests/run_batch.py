#!/usr/bin/env python3
"""Run a resumable shard of all three timing-test levels."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from timing_core import (
    DROProblem, METHOD_BLOCKS, annualised_sharpe, atomic_parquet, canonical_digest,
    cell_deltas, load_bundle, permutation_indices, run_path, runtime_metadata,
    sample_masks, sha256_file, validate_draws,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("portfolio-timing-tests"); logger.setLevel(logging.INFO)
    logger.handlers.clear(); formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="a"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def load_guard(bundle_dir: Path, manifest: dict, threads: int) -> tuple[dict, dict[str, np.ndarray]]:
    guard_path = bundle_dir / "runtime_guard.json"
    if not guard_path.exists():
        raise FileNotFoundError(f"{guard_path}\nRun validate_environment.py first.")
    guard = json.loads(guard_path.read_text())
    if guard["experiment_digest"] != manifest["experiment_digest"]:
        raise RuntimeError("runtime guard belongs to another experiment")
    current_runtime = runtime_metadata(threads)
    if canonical_digest(current_runtime) != guard["runtime_digest"]:
        raise RuntimeError("runtime guard does not match this host/environment/thread count")
    array_path = bundle_dir / guard["guard_array_file"]
    if sha256_file(array_path) != guard["guard_array_sha256"]:
        raise RuntimeError("runtime guard array SHA-256 mismatch")
    with np.load(array_path, allow_pickle=False) as loaded:
        paths = {name: loaded[name] for name in loaded.files}
    return guard, paths


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.stop <= args.start:
        raise ValueError("require 0 <= start < stop")
    if min(args.checkpoint_every, args.progress_every, args.threads) < 1:
        raise ValueError("checkpoint, progress and thread counts must be positive")
    if "MOSEK" not in cp.installed_solvers():
        raise RuntimeError("MOSEK is not installed or licensed")
    manifest, arrays = load_bundle(args.bundle_dir)
    if args.stop > manifest["draws"]:
        raise ValueError(f"stop exceeds declared draws={manifest['draws']}")
    guard, guard_paths = load_guard(args.bundle_dir, manifest, args.threads)
    log = logger_for(args.logs_dir / f"batch_{args.start:05d}_{args.stop:05d}.log")
    output_path = args.results_dir / (
        f"timing_tests_{args.start:05d}_{args.stop:05d}_{manifest['experiment_digest'][:12]}.parquet")
    log.info("starting batch [%d, %d)", args.start, args.stop)
    log.info("experiment_digest=%s", manifest["experiment_digest"])
    log.info("runtime_digest=%s | threads=%d", guard["runtime_digest"], args.threads)
    log.info("output=%s", output_path)
    expected = set(range(args.start, args.stop))
    if output_path.exists():
        output = pd.read_parquet(output_path)
        output = validate_draws(output, manifest, expected, require_complete=False)
        if not output["runtime_digest"].eq(guard["runtime_digest"]).all():
            raise RuntimeError("existing shard runtime mismatch")
        completed = set(output["draw"].astype(int)); log.info("resuming with %d/%d draws", len(completed), len(expected))
    else:
        output = pd.DataFrame(); completed: set[int] = set()
    static_paths = {code: {metric: guard_paths[f"{code}_static_{metric}"]
                           for metric in ("gross", "net", "turnover")} for code in ("L", "S")}
    masks = sample_masks(arrays)
    solver = DROProblem(manifest["window_rows"], manifest["n_assets"], args.threads)
    pending: list[dict] = []; started = time.time(); total = len(expected)
    for draw in range(args.start, args.stop):
        if draw in completed:
            continue
        indices: dict[str, np.ndarray] = {}; seeds: dict[str, int] = {}
        for method, block in manifest["method_blocks"].items():
            indices[method], seeds[method] = permutation_indices(
                manifest["n_months"], int(block), manifest["base_seed"], method,
                draw, manifest["experiment_digest"])
        # The simple monthly order is common to L and S: this is the dependence-preserving family draw.
        simple_paths = {code: run_path(
            solver, arrays, code, arrays[f"{code}_modulation"][indices["simple"]], manifest)
            for code in ("L", "S")}
        block_paths = {method: run_path(
            solver, arrays, "L", arrays["L_modulation"][indices[method]], manifest)
            for method in ("block6", "block12")}
        placebo_paths = {"simple": simple_paths["L"], **block_paths}
        row: dict[str, object] = {
            "draw": draw, "seed_simple": np.uint64(seeds["simple"]),
            "seed_block6": np.uint64(seeds["block6"]), "seed_block12": np.uint64(seeds["block12"]),
            "experiment_digest": manifest["experiment_digest"], "runtime_digest": guard["runtime_digest"],
        }
        for method, path in placebo_paths.items():
            row[f"L_{method}_sharpe_gross"] = annualised_sharpe(path["gross"])
            row[f"L_{method}_sharpe_net"] = annualised_sharpe(path["net"])
            row[f"L_{method}_mean_turnover"] = float(path["turnover"].mean())
        row["S_simple_mean_turnover"] = float(simple_paths["S"]["turnover"].mean())
        for basis in ("net", "gross"):
            cells = cell_deltas(static_paths, simple_paths, masks, basis)
            for cell in manifest["cells12"]:
                row[f"{basis}_{cell}"] = cells[cell]
            row[f"{basis}_maxT4"] = max(cells[cell] for cell in manifest["cells4"])
            row[f"{basis}_maxT12"] = max(cells[cell] for cell in manifest["cells12"])
        pending.append(row)
        if len(pending) >= args.checkpoint_every:
            output = pd.concat([output, pd.DataFrame(pending)], ignore_index=True)
            output = output.drop_duplicates("draw", keep="last").sort_values("draw")
            atomic_parquet(output, output_path); pending = []
        processed = len(output) + len(pending)
        if (draw - args.start + 1) % args.progress_every == 0:
            elapsed = time.time() - started; new_count = max(1, processed - len(completed))
            speed = elapsed / new_count; remaining = max(0, total - processed)
            log.info("draw=%d | progress=%d/%d (%.1f%%) | %.1fs/draw | ETA %.1f min | "
                     "maxT4=%+.6f maxT12=%+.6f", draw, processed, total,
                     100 * processed / total, speed, speed * remaining / 60,
                     row["net_maxT4"], row["net_maxT12"])
    if pending:
        output = pd.concat([output, pd.DataFrame(pending)], ignore_index=True)
    output = validate_draws(output, manifest, expected, require_complete=True)
    if not output["runtime_digest"].eq(guard["runtime_digest"]).all():
        raise RuntimeError("completed shard runtime mismatch")
    atomic_parquet(output, output_path)
    log.info("COMPLETE | rows=%d | elapsed=%.1f min", len(output), (time.time() - started) / 60)
    log.info("saved %s", output_path)


if __name__ == "__main__":
    main()
