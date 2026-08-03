"""Shared deterministic engine for the three-level portfolio timing tests."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
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


def load_bundle(bundle_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest_path = bundle_dir / "timing_test_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    claimed = manifest["experiment_digest"]
    contract = dict(manifest); contract.pop("experiment_digest")
    if canonical_digest(contract) != claimed:
        raise RuntimeError("experiment manifest digest mismatch")
    array_path = bundle_dir / manifest["array_file"]
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
