"""Shared deterministic engine for the three-level portfolio timing tests."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import time
from pathlib import Path

import cvxpy as cp
import mosek
import numpy as np
import pandas as pd


METHOD_BLOCKS = {"simple": 1, "block6": 6, "block12": 12}


def canonical_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def runtime_metadata(threads: int) -> dict[str, str]:
    return {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "cvxpy": cp.__version__,
        "mosek": ".".join(map(str, mosek.Env().getversion())), "threads": str(int(threads)),
    }


def load_bundle_files(manifest_path: Path, array_path: Path | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest["experiment_digest"]
    contract = dict(manifest); contract.pop("experiment_digest")
    if canonical_digest(contract) != claimed:
        raise RuntimeError("experiment manifest digest mismatch")
    if array_path is None:
        array_path = manifest_path.parent / manifest["array_file"]
    if sha256_file(array_path) != manifest["array_sha256"]:
        raise RuntimeError("input array SHA-256 mismatch")
    with np.load(array_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    expected = (manifest["n_months"], manifest["window_rows"], manifest["n_assets"])
    for code in ("L", "S"):
        if arrays[f"{code}_estimation_returns"].shape != expected:
            raise RuntimeError(f"{code}: estimation tensor shape mismatch")
        if arrays[f"{code}_permnos"].shape != (manifest["n_months"], manifest["n_assets"]):
            raise RuntimeError(f"{code}: PERMNO tensor shape mismatch")
    if not np.array_equal(arrays["L_return_date_ns"], arrays["S_return_date_ns"]):
        raise RuntimeError("large- and small-cap calendars differ")
    for block in manifest["method_blocks"].values():
        if manifest["n_months"] % int(block):
            raise RuntimeError(f"n_months is not divisible by block length {block}")
    return manifest, arrays


def load_bundle(bundle_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    return load_bundle_files(bundle_dir / "timing_test_manifest.json")


class DROProblem:
    """Parameterized direct-W2 SOCP used by ``04_R_Portfolio``.

    For a linear loss and a 2-Wasserstein ambiguity radius ``epsilon``,
    the problem is

        minimize    -mean(returns).T @ weights + epsilon * ||weights||_2
        subject to   sum(weights) = 1
                     weights >= 0.

    ``returns`` and ``epsilon`` are CVXPY parameters so the same compiled
    problem can be reused across dates and permutation draws.
    """

    def __init__(self, rows: int, assets: int, threads: int) -> None:
        self.returns = cp.Parameter((rows, assets))
        self.epsilon = cp.Parameter(nonneg=True)
        self.weights = cp.Variable(assets, nonneg=True)
        mean_returns = cp.sum(self.returns, axis=0) / rows
        self.problem = cp.Problem(
            cp.Minimize(
                -mean_returns @ self.weights
                + self.epsilon * cp.norm(self.weights, 2)
            ),
            [cp.sum(self.weights) == 1],
        )
        self.threads = int(threads)

    def solve(self, returns: np.ndarray, epsilon: float) -> np.ndarray:
        returns = np.asarray(returns, dtype=float)
        if returns.shape != self.returns.shape:
            raise ValueError(
                f"return matrix shape {returns.shape} != expected {self.returns.shape}"
            )
        if not np.isfinite(returns).all():
            raise ValueError("return matrix contains non-finite values")
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")

        self.returns.value = returns
        self.epsilon.value = float(epsilon)
        self.problem.solve(
            solver=cp.MOSEK,
            warm_start=False,
            mosek_params={"MSK_IPAR_NUM_THREADS": self.threads},
        )
        if self.weights.value is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MOSEK failure: {self.problem.status}")
        weights = np.clip(np.asarray(self.weights.value, dtype=float), 0.0, None)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise RuntimeError("invalid solver weights")
        weights /= weights.sum()
        if abs(weights.sum() - 1.0) > 1e-10:
            raise RuntimeError("weight reconciliation failure")
        return weights


def run_path(solver: DROProblem, arrays: dict[str, np.ndarray], code: str,
             modulation: np.ndarray, manifest: dict) -> dict[str, np.ndarray]:
    estimation = arrays[f"{code}_estimation_returns"]
    permnos = arrays[f"{code}_permnos"]; hold_factors = arrays[f"{code}_hold_factors"]
    delisted = arrays[f"{code}_delisted"]; cost_rate = arrays[f"{code}_cost_rate"]
    epsilon_base = float(arrays["epsilon_base"][0]); n_months = len(modulation)
    gross = np.empty(n_months); net = np.empty(n_months); turnover = np.empty(n_months)
    previous_risky: dict[int, float] = {}
    for month in range(n_months):
        epsilon = float(np.clip(epsilon_base * modulation[month],
                                manifest["epsilon_floor"], manifest["epsilon_cap"]))
        weights = solver.solve(estimation[month], epsilon)
        target = {int(p): float(w) for p, w in zip(permnos[month], weights)}
        union = set(previous_risky) | set(target)
        month_turnover = float(sum(abs(target.get(p, 0.0) - previous_risky.get(p, 0.0)) for p in union))
        terminal = weights * hold_factors[month]; gross_factor = float(terminal.sum())
        if not np.isfinite(gross_factor) or gross_factor <= 0:
            raise RuntimeError(f"{code} month {month}: non-positive gross factor")
        survives = ~delisted[month]
        previous_risky = {int(p): float(v / gross_factor)
                          for p, v, alive in zip(permnos[month], terminal, survives)
                          if alive and v > 0.0}
        end_cash = float(terminal[~survives].sum() / gross_factor)
        if abs(sum(previous_risky.values()) + end_cash - 1.0) > 1e-10:
            raise RuntimeError(f"{code} month {month}: risky/cash reconciliation failure")
        gross[month] = gross_factor - 1.0; turnover[month] = month_turnover
        net[month] = gross[month] - cost_rate[month] * month_turnover
    return {"gross": gross, "net": net, "turnover": turnover}


def annualised_sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float); sd = values.std(ddof=1)
    if len(values) < 2 or not np.isfinite(sd) or sd <= 0:
        raise RuntimeError("Sharpe is undefined")
    return float(values.mean() / sd * math.sqrt(12.0))


def sample_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(arrays["L_return_date_ns"])
    series = dates.to_series(index=np.arange(len(dates)))
    median_rho = float(np.median(arrays["L_sqrt_rho"]))
    return {
        "all": np.ones(len(dates), dtype=bool),
        "exdc": ~series.between("1999-01-01", "2001-12-31").to_numpy(),
        "d1": (dates >= "1995-01-01") & (dates <= "2004-12-31"),
        "d2": (dates >= "2005-01-01") & (dates <= "2014-12-31"),
        "d3": (dates >= "2015-01-01") & (dates <= "2025-12-31"),
        "rho_hi": arrays["L_sqrt_rho"] >= median_rho,
        "rho_lo": arrays["L_sqrt_rho"] < median_rho,
    }


def cell_deltas(static_paths: dict, dynamic_paths: dict, masks: dict, basis: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for code in ("L", "S"):
        for sample in ("all", "exdc", "d1", "d2", "d3"):
            mask = masks[sample]
            output[f"{code}_{sample}"] = (
                annualised_sharpe(dynamic_paths[code][basis][mask])
                - annualised_sharpe(static_paths[code][basis][mask]))
    for sample in ("rho_hi", "rho_lo"):
        mask = masks[sample]
        output[f"L_{sample}"] = (
            annualised_sharpe(dynamic_paths["L"][basis][mask])
            - annualised_sharpe(static_paths["L"][basis][mask]))
    return output


def permutation_indices(length: int, block: int, base_seed: int, method: str,
                        draw: int, experiment_digest: str) -> tuple[np.ndarray, int]:
    if length % block:
        raise ValueError("length must be divisible by block length")
    payload = f"{base_seed}|{method}|{draw}|{experiment_digest}".encode()
    seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    order = np.random.default_rng(seed).permutation(length // block)
    indices = np.concatenate([np.arange(b * block, (b + 1) * block) for b in order])
    if sorted(indices.tolist()) != list(range(length)):
        raise RuntimeError("invalid permutation indices")
    return indices, seed


def required_draw_columns(manifest: dict) -> set[str]:
    identity = {"draw", "seed_simple", "seed_block6", "seed_block12",
                "experiment_digest", "runtime_digest"}
    placebo = {f"L_{method}_{metric}" for method in METHOD_BLOCKS
               for metric in ("sharpe_gross", "sharpe_net", "mean_turnover")}
    cells = {f"{basis}_{cell}" for basis in ("net", "gross") for cell in manifest["cells12"]}
    maxima = {f"{basis}_maxT{family}" for basis in ("net", "gross") for family in (4, 12)}
    return identity | placebo | cells | maxima | {"S_simple_mean_turnover"}


def validate_draws(frame: pd.DataFrame, manifest: dict, expected_draws: set[int],
                   require_complete: bool = True) -> pd.DataFrame:
    missing_columns = required_draw_columns(manifest) - set(frame.columns)
    if missing_columns:
        raise RuntimeError(f"draw file missing columns: {sorted(missing_columns)}")
    frame = frame.drop_duplicates("draw", keep="last").sort_values("draw").reset_index(drop=True)
    actual = set(frame["draw"].astype(int))
    if require_complete and actual != expected_draws:
        raise RuntimeError(f"draw coverage mismatch; missing={sorted(expected_draws-actual)[:10]}")
    if actual - expected_draws:
        raise RuntimeError("draw file contains draws outside its declared range")
    if not frame["experiment_digest"].eq(manifest["experiment_digest"]).all():
        raise RuntimeError("draw experiment digest mismatch")
    for seed_column in ("seed_simple", "seed_block6", "seed_block12"):
        if frame[seed_column].duplicated().any():
            raise RuntimeError(f"duplicate seeds in {seed_column}")
    seed_matrix = frame[["seed_simple", "seed_block6", "seed_block12"]].to_numpy().ravel()
    if len(np.unique(seed_matrix)) != len(seed_matrix):
        raise RuntimeError("a seed is reused across placebo methods")
    statistic_columns = sorted(required_draw_columns(manifest) - {
        "draw", "seed_simple", "seed_block6", "seed_block12",
        "experiment_digest", "runtime_digest"})
    if not np.isfinite(frame[statistic_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("draw file contains a non-finite statistic")
    for basis in ("net", "gross"):
        max4 = frame[[f"{basis}_{c}" for c in manifest["cells4"]]].max(axis=1)
        max12 = frame[[f"{basis}_{c}" for c in manifest["cells12"]]].max(axis=1)
        if not np.allclose(max4, frame[f"{basis}_maxT4"], rtol=0, atol=1e-14):
            raise RuntimeError(f"{basis} maxT4 reconstruction failure")
        if not np.allclose(max12, frame[f"{basis}_maxT12"], rtol=0, atol=1e-14):
            raise RuntimeError(f"{basis} maxT12 reconstruction failure")
    return frame


def observed_statistics(manifest: dict, arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Observed real-path statistics from the signed exported arrays."""
    static = {code: {metric: arrays[f"{code}_expected_static_{metric}"]
                     for metric in ("gross", "net", "turnover")} for code in ("L", "S")}
    dynamic = {code: {metric: arrays[f"{code}_expected_dynamic_{metric}"]
                      for metric in ("gross", "net", "turnover")} for code in ("L", "S")}
    masks = sample_masks(arrays)
    cells = {basis: cell_deltas(static, dynamic, masks, basis) for basis in ("net", "gross")}
    for basis in ("net", "gross"):
        for cell in manifest["cells12"]:
            if abs(cells[basis][cell] - manifest[f"observed_cells_{basis}"][cell]) > 1e-12:
                raise RuntimeError(f"signed observed statistic mismatch: {basis} {cell}")
    real = {code: {strategy: {
        "sharpe_gross": annualised_sharpe((static if strategy == "static" else dynamic)[code]["gross"]),
        "sharpe_net": annualised_sharpe((static if strategy == "static" else dynamic)[code]["net"]),
        "mean_turnover": float((static if strategy == "static" else dynamic)[code]["turnover"].mean()),
    } for strategy in ("static", "dynamic")} for code in ("L", "S")}
    return cells, real


def load_or_compute(manifest_path: Path, array_path: Path, cache_path: Path,
                    threads: int = 1, checkpoint_every: int = 5,
                    progress_every: int = 5) -> tuple[dict, dict[str, np.ndarray], pd.DataFrame]:
    """Load the signed VPS panel or reconstruct all three levels locally and resumably."""
    manifest, arrays = load_bundle_files(manifest_path, array_path)
    draws = int(manifest["draws"]); expected = set(range(draws))
    local_runtime = canonical_digest(runtime_metadata(threads))
    if cache_path.exists():
        output = pd.read_parquet(cache_path)
        if len(output) == draws:
            return manifest, arrays, validate_draws(output, manifest, expected, require_complete=True)
        output = validate_draws(output, manifest, expected, require_complete=False)
        if not output["runtime_digest"].eq(local_runtime).all():
            raise RuntimeError("partial timing-test checkpoint belongs to another runtime")
    else:
        output = pd.DataFrame()
    completed = set(output.get("draw", pd.Series(dtype=int)).astype(int)); missing = sorted(expected - completed)
    if not missing:
        return manifest, arrays, validate_draws(output, manifest, expected, require_complete=True)
    if "MOSEK" not in cp.installed_solvers():
        raise RuntimeError("MOSEK is required to reconstruct missing timing-test draws")
    solver = DROProblem(manifest["window_rows"], manifest["n_assets"], threads)
    static_paths = {}
    for code in ("L", "S"):
        static_paths[code] = run_path(solver, arrays, code, np.ones(manifest["n_months"]), manifest)
        return_error = max(np.max(np.abs(
            static_paths[code][basis] - arrays[f"{code}_expected_static_{basis}"]))
            for basis in ("gross", "net"))
        turnover_error = np.max(np.abs(
            static_paths[code]["turnover"] - arrays[f"{code}_expected_static_turnover"]))
        if return_error > 1e-6 or turnover_error > 1e-5:
            raise RuntimeError(f"{code}: local numerical guard failed ({return_error:.3e}, {turnover_error:.3e})")
    masks = sample_masks(arrays); pending: list[dict] = []; started = time.time()
    print(f"timing tests: computing {len(missing):,} missing draws; checkpoint every {checkpoint_every}")
    for ordinal, draw in enumerate(missing, start=1):
        indices, seeds = {}, {}
        for method, block in manifest["method_blocks"].items():
            indices[method], seeds[method] = permutation_indices(
                manifest["n_months"], int(block), manifest["base_seed"], method,
                draw, manifest["experiment_digest"])
        simple = {code: run_path(
            solver, arrays, code, arrays[f"{code}_modulation"][indices["simple"]], manifest)
            for code in ("L", "S")}
        placebo = {"simple": simple["L"]}
        for method in ("block6", "block12"):
            placebo[method] = run_path(
                solver, arrays, "L", arrays["L_modulation"][indices[method]], manifest)
        row: dict[str, object] = {
            "draw": draw, "seed_simple": np.uint64(seeds["simple"]),
            "seed_block6": np.uint64(seeds["block6"]), "seed_block12": np.uint64(seeds["block12"]),
            "experiment_digest": manifest["experiment_digest"], "runtime_digest": local_runtime,
        }
        for method, path in placebo.items():
            row[f"L_{method}_sharpe_gross"] = annualised_sharpe(path["gross"])
            row[f"L_{method}_sharpe_net"] = annualised_sharpe(path["net"])
            row[f"L_{method}_mean_turnover"] = float(path["turnover"].mean())
        row["S_simple_mean_turnover"] = float(simple["S"]["turnover"].mean())
        for basis in ("net", "gross"):
            cells = cell_deltas(static_paths, simple, masks, basis)
            for cell in manifest["cells12"]:
                row[f"{basis}_{cell}"] = cells[cell]
            row[f"{basis}_maxT4"] = max(cells[c] for c in manifest["cells4"])
            row[f"{basis}_maxT12"] = max(cells[c] for c in manifest["cells12"])
        pending.append(row)
        if len(pending) >= checkpoint_every or ordinal == len(missing):
            output = pd.concat([output, pd.DataFrame(pending)], ignore_index=True)
            output = output.drop_duplicates("draw", keep="last").sort_values("draw")
            atomic_parquet(output, cache_path); pending = []
        if ordinal % progress_every == 0 or ordinal == len(missing):
            speed = (time.time() - started) / ordinal
            print(f"  {ordinal}/{len(missing)} | {speed:.1f}s/draw | "
                  f"ETA {speed * (len(missing)-ordinal) / 60:.1f} min")
    output = validate_draws(output, manifest, expected, require_complete=True)
    atomic_parquet(output, cache_path)
    return manifest, arrays, output


def corrected_p(null: pd.Series | np.ndarray, observed: float) -> tuple[int, float]:
    values = np.asarray(null, dtype=float); exceedances = int(np.count_nonzero(values >= observed))
    return exceedances, float((exceedances + 1) / (len(values) + 1))


def inference_tables(draws: pd.DataFrame, manifest: dict, arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive level-1 placebo and levels-2/3 family inference from one validated panel."""
    draws = validate_draws(draws, manifest, set(range(manifest["draws"])), require_complete=True)
    observed, real = observed_statistics(manifest, arrays)
    placebo_rows = []
    for method, block in manifest["method_blocks"].items():
        for basis in ("net", "gross"):
            value = real["L"]["dynamic"][f"sharpe_{basis}"]
            r, p = corrected_p(draws[f"L_{method}_sharpe_{basis}"], value)
            placebo_rows.append({"method": method, "block_months": int(block), "basis": basis,
                "observed_sharpe": value, "exceedances": r, "draws": len(draws), "p_value": p,
                "null_mean": float(draws[f"L_{method}_sharpe_{basis}"].mean()),
                "null_q95": float(draws[f"L_{method}_sharpe_{basis}"].quantile(.95)),
                "experiment_digest": manifest["experiment_digest"]})
    cell_rows = []
    for basis in ("net", "gross"):
        for cell in manifest["cells12"]:
            r_m, p_m = corrected_p(draws[f"{basis}_{cell}"], observed[basis][cell])
            r_12, p_12 = corrected_p(draws[f"{basis}_maxT12"], observed[basis][cell])
            r_4, p_4 = (corrected_p(draws[f"{basis}_maxT4"], observed[basis][cell])
                        if cell in manifest["cells4"] else (None, np.nan))
            cell_rows.append({"cell": cell, "basis": basis,
                "observed_delta_sharpe": observed[basis][cell], "marginal_exceedances": r_m,
                "p_marginal": p_m, "maxT4_exceedances": r_4, "p_maxT4": p_4,
                "maxT12_exceedances": r_12, "p_maxT12": p_12,
                "is_target": cell == manifest["target_cell"], "draws": len(draws),
                "experiment_digest": manifest["experiment_digest"]})
    placebo = pd.DataFrame(placebo_rows); inference = pd.DataFrame(cell_rows)
    target = inference[(inference["basis"] == manifest["primary_basis"]) & inference["is_target"]]
    if len(target) != 1 or float(target.iloc[0]["p_maxT12"]) < float(target.iloc[0]["p_maxT4"]):
        raise RuntimeError("target uniqueness or nested-family monotonicity failure")
    return placebo, inference
