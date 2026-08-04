"""Public, data-safe execution helpers for thesis result notebooks.

The full empirical notebooks require licensed CRSP inputs.  Public mode instead
verifies and materializes the frozen research outputs released under
``artifacts/``.  It never reads security-level returns, holdings, identifiers,
or portfolio weights.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display


FIGURES = {
    "signal": (
        "R_01_signal_overview.pdf",
        "R_02_event_study.pdf",
        "R_03_power_curves.pdf",
        "R_04_roc_family.pdf",
        "R_05_robust_M.pdf",
        "R_06_robust_lambda.pdf",
        "R_07_robust_scaling.pdf",
        "R_08_robust_scaling_exponent_ts.pdf",
        "R_09_robust_scaling_exponent.pdf",
        "R_10_robust_distance.pdf",
        "R_11_robust_barycenter.pdf",
        "R_12_vol_overlay.pdf",
    ),
    "portfolio": (
        "R_P_01_radius_time.pdf",
        "R_P_02_concentration_vs_eps.pdf",
        "R_P_03_relative_gain.pdf",
        "R_P_04_placebo_timing.pdf",
        "R_P_05_attribution.pdf",
        "R_P_06_landscape.pdf",
        "R_P_07_degeneracy.pdf",
        "R_P_08_universe_gain.pdf",
        "R_P_09_maxt_multiplicity.pdf",
        "R_P_10_useful_window.pdf",
        "R_P_11_robustness_levers.pdf",
        "R_P_12_decades_universe.pdf",
        "R_P_13_radius_forms_fullsample.pdf",
        "R_P_14_crises_signal_forms.pdf",
    ),
    "methodology": (
        "M_01_rescaling_aapl.pdf",
        "M_02_empirical_measure.pdf",
        "M_03_barycenter.pdf",
        "M_04_curse_dimensionality.pdf",
        "M_05_sliced_projection.pdf",
        "M_06_w2_quantile.pdf",
    ),
}

TABLE_GROUPS = {
    "signal": "signal",
    "portfolio": "portfolio",
    "methodology": "methodology",
}

EXTRA_FIGURES = {
    "methodology": (
        ("appendix", "B_01_cones.pdf"),
        ("appendix", "D_01_mc_convergence.pdf"),
    ),
}


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a notebook or the current directory."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "artifacts" / "SHA256SUMS").is_file():
            return candidate
    raise FileNotFoundError("repository root with artifacts/SHA256SUMS not found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(root: Path) -> int:
    """Verify every file covered by the public artifact checksum ledger."""
    ledger = root / "artifacts" / "SHA256SUMS"
    entries = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing release artifact: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch: {relative}")
        entries.append(relative)
    if not entries:
        raise AssertionError("empty artifacts/SHA256SUMS ledger")
    return len(entries)


def _validate_signal_data(root: Path) -> dict[str, object]:
    base = root / "artifacts" / "data" / "signal"
    signal = pd.read_parquet(base / "rho_pit_validated.parquet")
    required = {
        "formation_month", "universe", "config_id", "rho", "sqrt_rho",
        "engine_version", "config_digest",
    }
    if not required.issubset(signal.columns):
        raise AssertionError(f"signal release schema missing {sorted(required - set(signal.columns))}")
    if signal.empty or signal[["rho", "sqrt_rho"]].isna().any().any():
        raise AssertionError("invalid released signal panel")
    reference = signal.loc[
        signal["config_id"].astype(str).str.contains("reference", case=False, na=False)
    ]
    if reference.empty:
        reference = signal
    placebo = pd.read_parquet(base / "placebo_h0h1.parquet")
    power = pd.read_parquet(base / "power_curves.parquet")
    roc = pd.read_parquet(base / "roc_intensity.parquet")
    mc = pd.read_parquet(base / "mc_convergence.parquet")
    if len(placebo) != 300 or power.empty or roc.empty or mc.empty:
        raise AssertionError("incomplete public signal diagnostics")
    return {
        "signal_rows": len(signal),
        "universes": sorted(signal["universe"].astype(str).unique().tolist()),
        "configurations": int(signal["config_id"].nunique()),
        "formation_start": str(pd.to_datetime(signal["formation_month"]).min().date()),
        "formation_end": str(pd.to_datetime(signal["formation_month"]).max().date()),
        "placebo_windows": len(placebo),
        "power_rows": len(power),
        "roc_rows": len(roc),
        "mc_rows": len(mc),
    }


def _annualized_sharpe(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="raise")
    return float(np.sqrt(12.0) * series.mean() / series.std(ddof=1))


def _validate_portfolio_data(root: Path) -> dict[str, object]:
    base = root / "artifacts" / "data" / "portfolio"
    monthly = pd.read_parquet(base / "portfolio_monthly.parquet")
    required = {"strategy", "return_date", "gross_return", "net_return", "turnover"}
    if not required.issubset(monthly.columns):
        raise AssertionError(f"portfolio release schema missing {sorted(required - set(monthly.columns))}")
    if set(monthly["strategy"]) != {"static", "dynamic"}:
        raise AssertionError("unexpected strategy set in released monthly paths")
    if not monthly.groupby("strategy")["return_date"].nunique().eq(372).all():
        raise AssertionError("released monthly paths must contain 372 months per strategy")
    metrics = pd.read_parquet(base / "P_01_static_dynamic_metrics.parquet")
    reported = metrics.loc[metrics["basis"].eq("net")].set_index("strategy")["Sharpe"]
    rebuilt = monthly.groupby("strategy")["net_return"].apply(_annualized_sharpe)
    for strategy in ("static", "dynamic"):
        if not np.isclose(rebuilt[strategy], reported[strategy], rtol=0.0, atol=5e-12):
            raise AssertionError(f"released Sharpe reconstruction failed for {strategy}")
    inference = pd.read_parquet(base / "P_01_static_dynamic_inference.parquet")
    timing = pd.read_parquet(base / "P_07_P_08_timing_inference.parquet")
    manifest = json.loads((base / "portfolio_engine_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("engine_digest") != monthly["engine_digest"].iloc[0]:
        raise AssertionError("portfolio engine digest mismatch")
    net_delta = float(rebuilt["dynamic"] - rebuilt["static"])
    return {
        "months_per_strategy": 372,
        "static_net_sharpe": float(rebuilt["static"]),
        "dynamic_net_sharpe": float(rebuilt["dynamic"]),
        "delta_net_sharpe": net_delta,
        "hac_p_value": float(inference.loc[inference["basis"].eq("net"), "hac_p_value"].iloc[0]),
        "timing_cells": len(timing),
        "engine_digest": str(manifest["engine_digest"]),
    }


def _materialize(root: Path, section: str) -> tuple[int, int]:
    destination = root / "outputs" / "public" / section
    figures_out = destination / "images"
    tables_out = destination / "tables"
    figures_out.mkdir(parents=True, exist_ok=True)
    tables_out.mkdir(parents=True, exist_ok=True)

    figure_source = root / "artifacts" / "figures" / section
    figure_names = FIGURES[section]
    for name in figure_names:
        shutil.copy2(figure_source / name, figures_out / name)
    extra_figures = EXTRA_FIGURES.get(section, ())
    for group, name in extra_figures:
        shutil.copy2(root / "artifacts" / "figures" / group / name, figures_out / name)

    table_group = TABLE_GROUPS[section]
    table_source = root / "artifacts" / "tables" / table_group
    table_names = tuple(sorted(path.name for path in table_source.glob("*.tex")))
    for name in table_names:
        shutil.copy2(table_source / name, tables_out / name)
    return len(figure_names) + len(extra_figures), len(table_names)


def _display_links(root: Path, section: str) -> None:
    rows = ["| Artifact | Released file |", "|---|---|"]
    for name in FIGURES[section]:
        relative = Path("..") / "artifacts" / "figures" / section / name
        rows.append(f"| Figure | [{name}]({relative.as_posix()}) |")
    for group, name in EXTRA_FIGURES.get(section, ()):
        relative = Path("..") / "artifacts" / "figures" / group / name
        rows.append(f"| Figure | [{name}]({relative.as_posix()}) |")
    table_source = root / "artifacts" / "tables" / TABLE_GROUPS[section]
    for path in sorted(table_source.glob("*.tex")):
        relative = Path("..") / path.relative_to(root)
        rows.append(f"| Table | [{path.name}]({relative.as_posix()}) |")
    display(Markdown("\n".join(rows)))


def run_public_notebook(section: str, root: str | Path | None = None) -> dict[str, object]:
    """Validate and materialize one public result-notebook release.

    Parameters
    ----------
    section:
        One of ``signal``, ``portfolio`` or ``methodology``.
    root:
        Optional repository root. It is discovered automatically by default.
    """
    if section not in FIGURES:
        raise ValueError(f"unknown public notebook section: {section}")
    repo = find_repo_root(root)
    checked = verify_release(repo)
    if section == "signal":
        summary = _validate_signal_data(repo)
    elif section == "portfolio":
        summary = _validate_portfolio_data(repo)
    else:
        summary = {"data_dependency": "none in public mode"}
    n_figures, n_tables = _materialize(repo, section)
    summary.update(
        {
            "verified_release_files": checked,
            "materialized_figures": n_figures,
            "materialized_tables": n_tables,
            "output_directory": str(repo / "outputs" / "public" / section),
        }
    )
    display(Markdown(f"## Public release verification — {section.title()}"))
    display(pd.Series(summary, name="value").to_frame())
    _display_links(repo, section)
    print(f"PUBLIC MODE PASS — {section} | figures={n_figures} tables={n_tables}")
    return summary
