#!/usr/bin/env python3
"""Prepare the signed timing-test bundle from the corrected project inputs.

This is a local preparation utility.  It is not called by 04_R_Portfolio and
is not used during the VPS permutation run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import polars as pl

from timing_core import (
    DROProblem,
    atomic_json,
    canonical_digest,
    cell_deltas,
    load_bundle,
    run_path,
    sample_masks,
    sha256_file,
)


W_EST = 36
EPS_FLOOR = 1e-4
EPS_CAP = 2.0
BETA_EK = 0.10
EPS_EK = float(np.clip(math.sqrt(2.0 * math.log(1.0 / BETA_EK) / W_EST),
                       EPS_FLOOR, EPS_CAP))
DRAWS = 1000
BASE_SEED = 20250318
CROSS_RETURN_TOLERANCE = 1e-6
# The notebook rebuilds each CVXPY problem, whereas the standalone reuses a
# parameterised problem. MOSEK can therefore differ by a few 1e-5 in turnover
# while producing economically identical return paths.
CROSS_TURNOVER_TOLERANCE = 5e-5
METHOD_BLOCKS = {"block12": 12, "block6": 6, "simple": 1}
CELLS4 = ["L_all", "L_exdc", "S_all", "S_exdc"]
CELLS12 = [
    "L_all", "L_exdc", "S_all", "S_exdc",
    "L_d1", "L_d2", "L_d3", "S_d1", "S_d2", "S_d3",
    "L_rho_hi", "L_rho_lo",
]
BIG_NULLS = {(76614, "2015-06-09")}
SMALL_NULLS = {(25312, "1997-09-22")} | {
    (63079, f"2005-04-{day:02d}")
    for day in (8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--threads", type=int, default=1)
    return parser.parse_args()


def cost_bps(date) -> float:
    year = pd.Timestamp(date).year
    if year <= 1999:
        return 50.0
    if year <= 2009:
        return 30.0
    return 20.0


def ns(values) -> np.ndarray:
    return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def source_digest(paths: list[Path]) -> str:
    return canonical_digest({str(path.resolve()): sha256_file(path) for path in paths})


def build_universe(
    code: str,
    panel: Path,
    signal: Path,
    allowed_nulls: set[tuple[int, str]],
) -> dict[str, np.ndarray]:
    print(f"preparing {code}: {panel.name}", flush=True)
    scan = pl.scan_parquet(panel)

    members = (
        scan.filter(pl.col("is_formation_member") == 1)
        .select([
            pl.col("formation_member_month").alias("formation_month"),
            pl.col("formation_member_active_month").alias("holding_month"),
            "PERMNO",
            pl.col("formation_member_rank").alias("rank"),
            pl.col("DlyCalDt").alias("formation_date"),
            pl.col("formation_member_strict_hist60").alias("strict60"),
        ])
        .collect(engine="streaming")
        .to_pandas()
    )
    for column in ("formation_month", "holding_month", "formation_date"):
        members[column] = pd.to_datetime(members[column])
    members["PERMNO"] = members["PERMNO"].astype(int)
    members = members.sort_values(["formation_month", "rank"]).reset_index(drop=True)
    grouped = members.groupby("formation_month", sort=True)
    assert grouped.size().eq(100).all()
    assert grouped["PERMNO"].nunique().eq(100).all()
    assert grouped["rank"].apply(
        lambda values: sorted(values.tolist()) == list(range(1, 101))
    ).all()
    assert grouped["formation_date"].nunique().eq(1).all()
    assert members["strict60"].all()
    assert members["holding_month"].eq(
        members["formation_month"] + pd.offsets.MonthBegin(1)
    ).all()

    monthly_long = (
        scan.select(["PERMNO", "DlyCalDt", "DlyRet"])
        .with_columns(pl.col("DlyCalDt").dt.month_end().alias("month_end"))
        .group_by(["month_end", "PERMNO"])
        .agg([
            pl.col("DlyRet").count().alias("n_valid"),
            (pl.col("DlyRet") + 1.0).product().alias("factor"),
        ])
        .with_columns(
            pl.when(pl.col("n_valid") >= 15)
            .then(pl.col("factor") - 1.0)
            .otherwise(None)
            .alias("monthly_return")
        )
        .select(["month_end", "PERMNO", "monthly_return"])
        .collect(engine="streaming")
        .to_pandas()
    )
    monthly = monthly_long.pivot(
        index="month_end", columns="PERMNO", values="monthly_return"
    ).sort_index()
    monthly.index = pd.to_datetime(monthly.index)
    monthly.columns = monthly.columns.astype(int)

    ledger: list[dict] = []
    for fm, frame in grouped:
        frame = frame.sort_values("rank")
        assets = frame["PERMNO"].astype(int).tolist()
        formation_month = pd.Timestamp(fm)
        end = formation_month + pd.offsets.MonthEnd(0)
        start = (
            formation_month - pd.DateOffset(months=W_EST - 1)
        ) + pd.offsets.MonthEnd(0)
        if start < monthly.index.min():
            continue
        estimation = monthly.loc[start:end, assets]
        assert estimation.shape == (W_EST, 100), (
            f"{code} {formation_month:%Y-%m}: {estimation.shape}"
        )
        assert estimation.notna().all().all(), (
            f"{code} {formation_month:%Y-%m}: NaN in estimation matrix"
        )
        ledger.append({
            "formation_month": formation_month,
            "formation_date": pd.Timestamp(frame["formation_date"].iloc[0]),
            "holding_month": pd.Timestamp(frame["holding_month"].iloc[0]),
            "assets": assets,
            "estimation": estimation.to_numpy(dtype=np.float64),
        })
    assert len(ledger) == 373

    panel_last_month = (
        scan.select(pl.col("DlyCalDt").max())
        .collect(engine="streaming")
        .item()
    )
    panel_last_period = pd.Timestamp(panel_last_month).to_period("M")
    realised = [
        entry for entry in ledger
        if entry["holding_month"].to_period("M") <= panel_last_period
    ]
    assert len(realised) == 372 and len(ledger) - len(realised) == 1

    pair_rows = []
    for entry in realised:
        month_end = entry["holding_month"] + pd.offsets.MonthEnd(0)
        for rank, permno in enumerate(entry["assets"], start=1):
            pair_rows.append({
                "formation_month": entry["formation_month"].to_pydatetime(),
                "month_end": month_end.to_pydatetime(),
                "PERMNO": int(permno),
                "rank": rank,
            })
    pairs = pl.DataFrame(pair_rows)
    held = (
        scan.select(["PERMNO", "DlyCalDt", "DlyRet", "is_delisting_event"])
        .with_columns(pl.col("DlyCalDt").dt.month_end().alias("month_end"))
        .join(pairs.lazy(), on=["PERMNO", "month_end"], how="inner")
        .collect(engine="streaming")
    )
    null_rows = held.filter(pl.col("DlyRet").is_null())
    actual_nulls = {
        (int(permno), pd.Timestamp(date).strftime("%Y-%m-%d"))
        for permno, date in null_rows.select(["PERMNO", "DlyCalDt"]).iter_rows()
    }
    assert actual_nulls == allowed_nulls, (
        f"{code} held-day null set mismatch: {actual_nulls ^ allowed_nulls}"
    )
    assert not null_rows["is_delisting_event"].any()
    held = held.with_columns(pl.col("DlyRet").fill_null(0.0))
    market_days = held.group_by("month_end").agg(
        pl.col("DlyCalDt").n_unique().alias("n_market_days")
    )
    factors = (
        held.group_by(["formation_month", "month_end", "PERMNO", "rank"])
        .agg([
            (pl.col("DlyRet") + 1.0).product().alias("hold_factor"),
            pl.len().alias("n_days"),
            pl.col("DlyCalDt").max().alias("last_observed_date"),
            pl.col("DlyCalDt").filter(pl.col("is_delisting_event")).max().alias("event_date"),
            pl.col("is_delisting_event").any().alias("delisted"),
        ])
        .join(market_days, on="month_end", how="left")
        .sort(["formation_month", "rank"])
        .to_pandas()
    )
    assert factors.groupby("formation_month").size().eq(100).all()
    incomplete = factors[
        (~factors["delisted"]) & (factors["n_days"] != factors["n_market_days"])
    ]
    bad_delisting = factors[
        factors["delisted"]
        & (factors["last_observed_date"] != factors["event_date"])
    ]
    assert incomplete.empty
    assert bad_delisting.empty
    factor_groups = {
        pd.Timestamp(fm): frame.sort_values("rank")
        for fm, frame in factors.groupby("formation_month", sort=True)
    }

    rho = pd.read_parquet(signal).set_index("date")["rho"].sort_index()
    rho.index = pd.to_datetime(rho.index) + pd.offsets.MonthEnd(0)
    signal_rows = []
    for entry in ledger:
        date = entry["formation_month"] + pd.offsets.MonthEnd(0)
        assert date in rho.index
        value = float(rho.loc[date])
        signal_rows.append({
            "formation_month": entry["formation_month"],
            "sqrt_rho": math.sqrt(value),
        })
    timing = pd.DataFrame(signal_rows).sort_values("formation_month")
    timing["expanding_mean"] = timing["sqrt_rho"].expanding().mean()
    timing["modulation"] = timing["sqrt_rho"] / timing["expanding_mean"]
    timing = timing.set_index("formation_month")

    formation_months, formation_dates, holding_months, return_dates = [], [], [], []
    estimations, permnos, hold_factors, delisted = [], [], [], []
    modulation, sqrt_rho, cost_rate = [], [], []
    for entry in realised:
        fm = entry["formation_month"]
        legs = factor_groups[fm]
        assert legs["PERMNO"].astype(int).tolist() == entry["assets"]
        formation_months.append(fm)
        formation_dates.append(entry["formation_date"])
        holding_months.append(entry["holding_month"])
        return_dates.append(entry["holding_month"] + pd.offsets.MonthEnd(0))
        estimations.append(entry["estimation"])
        permnos.append(np.asarray(entry["assets"], dtype=np.int64))
        hold_factors.append(legs["hold_factor"].to_numpy(dtype=np.float64))
        delisted.append(legs["delisted"].to_numpy(dtype=bool))
        modulation.append(float(timing.loc[fm, "modulation"]))
        sqrt_rho.append(float(timing.loc[fm, "sqrt_rho"]))
        cost_rate.append(cost_bps(entry["formation_date"]) / 1e4)

    arrays = {
        f"{code}_formation_month_ns": ns(formation_months),
        f"{code}_formation_date_ns": ns(formation_dates),
        f"{code}_holding_month_ns": ns(holding_months),
        f"{code}_return_date_ns": ns(return_dates),
        f"{code}_estimation_returns": np.stack(estimations),
        f"{code}_permnos": np.stack(permnos),
        f"{code}_hold_factors": np.stack(hold_factors),
        f"{code}_delisted": np.stack(delisted),
        f"{code}_modulation": np.asarray(modulation, dtype=np.float64),
        f"{code}_sqrt_rho": np.asarray(sqrt_rho, dtype=np.float64),
        f"{code}_cost_rate": np.asarray(cost_rate, dtype=np.float64),
    }
    assert arrays[f"{code}_estimation_returns"].shape == (372, 36, 100)
    assert arrays[f"{code}_permnos"].shape == (372, 100)
    assert np.isfinite(arrays[f"{code}_estimation_returns"]).all()
    assert np.isfinite(arrays[f"{code}_hold_factors"]).all()
    return arrays


def clear_generated(bundle: Path) -> None:
    for folder_name in ("data", "results", "logs"):
        folder = bundle / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    pycache = bundle / "__pycache__"
    if pycache.exists():
        shutil.rmtree(pycache)


def write_checksums(bundle: Path) -> None:
    relative = [
        Path("README.md"), Path("environment.yml"), Path("timing_core.py"),
        Path("prepare_inputs.py"), Path("validate_environment.py"),
        Path("run_batch.py"), Path("merge_results.py"), Path("status.py"),
        Path("data/timing_test_manifest.json"),
        Path("data/timing_test_inputs.npz"),
    ]
    text = "".join(
        f"{sha256_file(bundle / path)}  {path}\n" for path in relative
    )
    (bundle / "SHA256SUMS").write_text(text)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    bundle = Path(__file__).resolve().parent
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if "MOSEK" not in cp.installed_solvers():
        raise RuntimeError("MOSEK is required")

    big_panel = root / "data/processed/nyse_big_caps_pit_daily.parquet"
    small_panel = root / "data/processed/nyse_small_caps_p20_p50_pit_daily.parquet"
    big_signal = root / "data/signals/V_uni.parquet"
    small_signal = root / "data/signals/V_small_uni.parquet"
    big_monthly = root / "outputs/results/portfolio/audit/portfolio_monthly.parquet"
    required = [big_panel, small_panel, big_signal, small_signal, big_monthly]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    arrays: dict[str, np.ndarray] = {}
    arrays.update(build_universe("L", big_panel, big_signal, BIG_NULLS))
    arrays.update(build_universe("S", small_panel, small_signal, SMALL_NULLS))
    arrays["epsilon_base"] = np.asarray([EPS_EK], dtype=np.float64)
    assert np.array_equal(arrays["L_return_date_ns"], arrays["S_return_date_ns"])

    preliminary = {
        "n_months": 372, "window_rows": 36, "n_assets": 100,
        "epsilon_floor": EPS_FLOOR, "epsilon_cap": EPS_CAP,
    }
    solver = DROProblem(36, 100, args.threads)
    solved: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for code in ("L", "S"):
        solved[code] = {}
        for strategy, path_modulation in (
            ("static", np.ones(372)),
            ("dynamic", arrays[f"{code}_modulation"]),
        ):
            print(f"solving observed {code} {strategy}", flush=True)
            path = run_path(solver, arrays, code, path_modulation, preliminary)
            solved[code][strategy] = path
            for metric, values in path.items():
                arrays[f"{code}_expected_{strategy}_{metric}"] = values

    # Independent cross-contract against the big-cap engine already produced by 04.
    locked = pd.read_parquet(big_monthly)
    locked["return_date"] = pd.to_datetime(locked["return_date"])
    for strategy in ("static", "dynamic"):
        frame = locked.loc[locked["strategy"].eq(strategy)].sort_values("return_date")
        assert len(frame) == 372
        return_error = max(
            np.max(np.abs(
                arrays[f"L_expected_{strategy}_{metric}"]
                - frame[column].to_numpy(dtype=float)
            ))
            for metric, column in (("gross", "gross_return"), ("net", "net_return"))
        )
        turnover_error = np.max(np.abs(
            arrays[f"L_expected_{strategy}_turnover"]
            - frame["turnover"].to_numpy(dtype=float)
        ))
        if (
            return_error > CROSS_RETURN_TOLERANCE
            or turnover_error > CROSS_TURNOVER_TOLERANCE
        ):
            raise RuntimeError(
                f"04 cross-contract failure {strategy}: "
                f"return={return_error:.3e}, turnover={turnover_error:.3e}"
            )
        print(
            f"PASS 04 cross-contract {strategy}: "
            f"return={return_error:.3e}, turnover={turnover_error:.3e}",
            flush=True,
        )

    clear_generated(bundle)
    input_path = bundle / "data/timing_test_inputs.npz"
    with input_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    array_sha = sha256_file(input_path)
    upstream = source_digest([big_panel, small_panel, big_signal, small_signal])
    static_paths = {code: solved[code]["static"] for code in ("L", "S")}
    dynamic_paths = {code: solved[code]["dynamic"] for code in ("L", "S")}
    masks = sample_masks(arrays)
    observed = {
        basis: cell_deltas(static_paths, dynamic_paths, masks, basis)
        for basis in ("net", "gross")
    }
    contract = {
        "schema_version": "portfolio-timing-three-level-v2.0.0",
        "source_bundle_digest": upstream,
        "array_file": "timing_test_inputs.npz",
        "array_sha256": array_sha,
        "draws": DRAWS,
        "base_seed": BASE_SEED,
        "n_months": 372,
        "window_rows": 36,
        "n_assets": 100,
        "epsilon_floor": EPS_FLOOR,
        "epsilon_cap": EPS_CAP,
        "primary_basis": "net",
        "target_cell": "S_exdc",
        "method_blocks": METHOD_BLOCKS,
        "cells4": CELLS4,
        "cells12": CELLS12,
        "observed_cells_net": {
            cell: float(observed["net"][cell]) for cell in CELLS12
        },
        "observed_cells_gross": {
            cell: float(observed["gross"][cell]) for cell in CELLS12
        },
        "date_contract": {
            "start": "1995-01-31",
            "end": "2025-12-31",
            "dotcom_exclusion": ["1999-01-01", "2001-12-31"],
            "decades": {
                "d1": ["1995-01-01", "2004-12-31"],
                "d2": ["2005-01-01", "2014-12-31"],
                "d3": ["2015-01-01", "2025-12-31"],
            },
            "rho_regimes": (
                "large-cap formation sqrt(rho), full-sample median, high >= median"
            ),
        },
        "dependence_contract": {
            "simple": "one monthly permutation shared by L and S within every draw",
            "block6": "separate six-month null for the level-1 large-cap placebo",
            "block12": "separate twelve-month null for the level-1 large-cap placebo",
        },
        "accounting": {
            "delisting": "terminal proceeds become cash until next rebalance",
            "initial_state": "100pct_cash",
            "net_return": "gross_return_minus_cost_rate_times_turnover",
            "turnover": "sum_abs_risky_assets_cash_excluded",
        },
        "solver_contract": {
            "solver": "MOSEK", "warm_start": False,
            "silent_fallback": False, "threads_runtime_guarded": True,
            "portfolio_engine": "portfolio-pit-v2.0.0-direct-w2-socp",
            "formulation": "linear-loss-direct-W2-SOCP",
            "notebook_cross_contract_tolerances": {
                "return": CROSS_RETURN_TOLERANCE,
                "turnover": CROSS_TURNOVER_TOLERANCE,
            },
        },
        "test_levels": {
            "level1": {
                "nulls": ["simple", "block6", "block12"],
                "statistic": "large-cap dynamic annualised Sharpe",
            },
            "level2": {
                "family": "cells4", "resampling": "simple",
                "statistic": "dynamic-minus-static annualised Sharpe",
                "target": "S_exdc",
            },
            "level3": {
                "family": "cells12", "resampling": "simple",
                "statistic": "dynamic-minus-static annualised Sharpe",
                "target": "S_exdc",
            },
        },
    }
    manifest = dict(contract, experiment_digest=canonical_digest(contract))
    atomic_json(manifest, bundle / "data/timing_test_manifest.json")
    write_checksums(bundle)
    loaded_manifest, _ = load_bundle(bundle / "data")
    assert loaded_manifest["experiment_digest"] == manifest["experiment_digest"]
    print("\nSIGNED STANDALONE INPUT READY", flush=True)
    print(f"digest={manifest['experiment_digest']}", flush=True)
    print(f"inputs={input_path}", flush=True)
    print(f"manifest={bundle / 'data/timing_test_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
