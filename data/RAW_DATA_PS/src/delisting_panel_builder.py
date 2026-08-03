"""Build reproducible CRSP daily panels with delisting-return adjustments.

This module preserves the realised missing-delisting-return imputations
embedded in the canonical 1990--2025 processed panel and provides the active,
reproducible reconstruction path through three explicit modes:

``reference``
    Reproduce the thesis panel exactly from the private reference ledger.  Any
    missing or unexpected historical key is a hard failure.  No random number
    is generated.

``seeded``
    Generate every missing event-day ``DelRet`` from a user-selected base seed.
    Per-observation seeds are derived from SHA-256, so results are independent
    of process, machine, row order and batch partition.

``extend``
    Give the historical reference ledger priority and use the SHA-256 seeded
    rule only for genuinely new events absent from the ledger.

The private ledger is CRSP-derived and must not be committed to a public
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA_VERSION = "delret-reference-ledger-v1"
BASE_SEED = 42

SRC_DIR = Path(__file__).resolve().parent
RAW_DIR = SRC_DIR.parent
DATA_DIR = RAW_DIR.parent
PRIVATE_DIR = RAW_DIR / "private"

DEFAULT_SOURCE = DATA_DIR / "crsp_daily_delistings_unprocessed.parquet"
DEFAULT_PROCESSED = DATA_DIR / "crsp_daily_shumway_delisting_processed.parquet"
DEFAULT_LEDGER = PRIVATE_DIR / "delret_reference_ledger_v1.parquet"
DEFAULT_MANIFEST = PRIVATE_DIR / "delret_reference_manifest_v1.json"

KEY_COLUMNS = ["PERMNO", "DlyCalDt"]

DELISTING_MAPPING = {
    "BKPY": "bankruptcy",
    "FING": "bankruptcy",
    "DELQ": "noncomp",
    "LP": "noncomp",
    "INSC": "noncomp",
    "CORQ": "merger",
    "MVOT": "exchange",
    "UNAV": "unknown",
}

IMPUTATION_CONFIG = {
    "bankruptcy": {"std": 0.15, "min": -0.95, "max": -0.20},
    "noncomp": {"std": 0.10, "min": -0.60, "max": -0.05},
    "merger": {"value": 0.0},
    "exchange": {"value": 0.0},
    "unknown": {"value": 0.0},
}

STOCHASTIC_REASONS = {"BKPY", "FING", "DELQ", "LP", "INSC"}

SOURCE_COLUMNS = [
    "PERMNO",
    "DlyCalDt",
    "DelistingDt",
    "DelRet",
    "DelReasonType",
    "PrimaryExch",
    "DlyRet",
]

PROCESSED_COLUMNS = [
    "PERMNO",
    "DlyCalDt",
    "DelRet",
    "DlyRet",
    "delist_category",
]


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(permno: int, date: Any, base_seed: int = BASE_SEED) -> int:
    """Derive a cross-process per-observation seed from SHA-256."""
    date_key = pd.Timestamp(date).strftime("%Y-%m-%d")
    stable_id = f"{int(permno)}_{date_key}"
    digest = hashlib.sha256(stable_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big") % 100_000
    return int(base_seed + offset)


def seeded_imputation(
    *,
    permno: int,
    date: Any,
    reason: Any,
    exchange: Any,
    base_seed: int = BASE_SEED,
) -> float:
    """Impute one observation with the documented SHA-256 seeded rule."""
    category = DELISTING_MAPPING.get(reason)
    config = IMPUTATION_CONFIG.get(category, {"value": 0.0})

    if "value" in config:
        return float(config["value"])

    mu = -0.30 if exchange in {"N", "A"} else -0.55
    seed = stable_seed(permno, date, base_seed=base_seed)
    rng = np.random.RandomState(seed)
    value = rng.normal(mu, config["std"])
    return float(np.clip(value, config["min"], config["max"]))


def _parquet_contract(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": parquet.metadata.num_columns,
        "created_by": parquet.metadata.created_by,
        "sha256": sha256_file(path),
    }


def _assert_aligned_row_group(
    source: pd.DataFrame,
    processed: pd.DataFrame,
    row_group: int,
) -> None:
    if len(source) != len(processed):
        raise AssertionError(f"row group {row_group}: row-count mismatch")

    source_permno = source["PERMNO"].to_numpy()
    processed_permno = processed["PERMNO"].to_numpy()
    if not np.array_equal(source_permno, processed_permno, equal_nan=True):
        raise AssertionError(f"row group {row_group}: PERMNO ordering mismatch")

    source_date = pd.to_datetime(source["DlyCalDt"]).to_numpy()
    processed_date = pd.to_datetime(processed["DlyCalDt"]).to_numpy()
    if not np.array_equal(source_date, processed_date):
        raise AssertionError(f"row group {row_group}: date ordering mismatch")


def extract_historical_ledger(
    source_path: Path,
    processed_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract the realised imputations from aligned source/processed panels."""
    source_file = pq.ParquetFile(source_path)
    processed_file = pq.ParquetFile(processed_path)

    if source_file.metadata.num_rows != processed_file.metadata.num_rows:
        raise AssertionError("source and processed panels have different row counts")
    if source_file.metadata.num_row_groups != processed_file.metadata.num_row_groups:
        raise AssertionError("source and processed panels have different row groups")

    ledger_parts: list[pd.DataFrame] = []
    max_return_error = 0.0

    for row_group in range(source_file.metadata.num_row_groups):
        source = source_file.read_row_group(
            row_group,
            columns=SOURCE_COLUMNS,
        ).to_pandas()
        processed = processed_file.read_row_group(
            row_group,
            columns=PROCESSED_COLUMNS,
        ).to_pandas()

        _assert_aligned_row_group(source, processed, row_group)

        source["DlyCalDt"] = pd.to_datetime(source["DlyCalDt"])
        source["DelistingDt"] = pd.to_datetime(source["DelistingDt"])
        processed["DlyCalDt"] = pd.to_datetime(processed["DlyCalDt"])

        event_day = (
            source["DelistingDt"].notna()
            & (source["DlyCalDt"] == source["DelistingDt"])
        )
        needs_imputation = event_day & source["DelRet"].isna()
        if not needs_imputation.any():
            continue

        offsets = np.flatnonzero(needs_imputation.to_numpy())
        selected_source = source.loc[
            needs_imputation,
            [
                "PERMNO",
                "DlyCalDt",
                "DelistingDt",
                "DelReasonType",
                "PrimaryExch",
                "DlyRet",
            ],
        ].copy()
        selected_processed = processed.loc[
            needs_imputation,
            ["DelRet", "DlyRet", "delist_category"],
        ].copy()

        ledger_part = pd.DataFrame(
            {
                "PERMNO": selected_source["PERMNO"].astype("int64").to_numpy(),
                "DlyCalDt": selected_source["DlyCalDt"].to_numpy(),
                "DelistingDt": selected_source["DelistingDt"].to_numpy(),
                "DelReasonType": selected_source["DelReasonType"].to_numpy(),
                "PrimaryExch": selected_source["PrimaryExch"].to_numpy(),
                "delist_category": selected_processed[
                    "delist_category"
                ].to_numpy(),
                "imputation_method": np.where(
                    selected_source["DelReasonType"].isin(STOCHASTIC_REASONS),
                    "stochastic_legacy",
                    "deterministic_zero",
                ),
                "DelRet_imputed": selected_processed["DelRet"]
                .astype("float64")
                .to_numpy(),
                "DlyRet_raw": selected_source["DlyRet"]
                .astype("float64")
                .to_numpy(),
                "DlyRet_adjusted": selected_processed["DlyRet"]
                .astype("float64")
                .to_numpy(),
                "source_row_group": np.full(
                    len(selected_source),
                    row_group,
                    dtype=np.int32,
                ),
                "source_row_offset": offsets.astype(np.int32),
            }
        )

        expected_adjusted = (
            (1.0 + ledger_part["DlyRet_raw"].fillna(0.0))
            * (1.0 + ledger_part["DelRet_imputed"].fillna(0.0))
            - 1.0
        )
        errors = np.abs(
            expected_adjusted.to_numpy()
            - ledger_part["DlyRet_adjusted"].to_numpy()
        )
        finite_errors = errors[np.isfinite(errors)]
        if finite_errors.size:
            max_return_error = max(
                max_return_error,
                float(finite_errors.max()),
            )
        if not np.allclose(
            expected_adjusted.to_numpy(),
            ledger_part["DlyRet_adjusted"].to_numpy(),
            rtol=0.0,
            atol=2e-15,
            equal_nan=True,
        ):
            raise AssertionError(
                f"row group {row_group}: adjusted-return reconstruction failed"
            )

        ledger_parts.append(ledger_part)

    if not ledger_parts:
        raise AssertionError("no historical imputations were found")

    ledger = pd.concat(ledger_parts, ignore_index=True)
    ledger = ledger.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)

    if ledger.duplicated(KEY_COLUMNS).any():
        duplicates = ledger.loc[
            ledger.duplicated(KEY_COLUMNS, keep=False),
            KEY_COLUMNS,
        ]
        raise AssertionError(f"ledger keys are not unique:\n{duplicates.head()}")
    if ledger["DelRet_imputed"].isna().any():
        raise AssertionError("ledger contains missing realised DelRet values")

    deterministic = ledger["imputation_method"].eq("deterministic_zero")
    if not np.allclose(
        ledger.loc[deterministic, "DelRet_imputed"].to_numpy(),
        0.0,
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("deterministic-zero ledger rows are not zero")

    diagnostics = {
        "rows": int(len(ledger)),
        "unique_keys": int(ledger[KEY_COLUMNS].drop_duplicates().shape[0]),
        "unique_permnos": int(ledger["PERMNO"].nunique()),
        "stochastic_legacy": int(
            ledger["imputation_method"].eq("stochastic_legacy").sum()
        ),
        "deterministic_zero": int(deterministic.sum()),
        "max_adjusted_return_error": max_return_error,
    }
    return ledger, diagnostics


def build_ledger(
    *,
    source_path: Path,
    processed_path: Path,
    ledger_path: Path,
    manifest_path: Path,
) -> None:
    """Create the private ledger and its cryptographic manifest."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ledger, diagnostics = extract_historical_ledger(
        source_path,
        processed_path,
    )

    temporary_ledger = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    table = pa.Table.from_pandas(ledger, preserve_index=False)
    pq.write_table(table, temporary_ledger, compression="zstd")
    os.replace(temporary_ledger, ledger_path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Exact private record of the realised missing event-day CRSP "
            "delisting-return imputations embedded in the reference 1990--2025 "
            "processed panel."
        ),
        "historical_producer": (
            "tmp/RAW_DATA_PS_legacy/"
            "01_crsp+del_processing_historical.ipynb"
        ),
        "canonical_entrypoint": "01_crsp+del_processing.ipynb",
        "historical_hash_note": (
            "The historical producer used Python hash() to derive per-row "
            "NumPy seeds.  The process hash secret was not retained.  The "
            "ledger values, not reverse-engineered seeds, are authoritative."
        ),
        "seeded_rule": {
            "base_seed": BASE_SEED,
            "key": "PERMNO_YYYY-MM-DD",
            "seed_derivation": (
                "BASE_SEED + int.from_bytes(SHA256(key)[:8], 'big') % 100000"
            ),
            "applies_to_reference_panel": False,
        },
        "source": _parquet_contract(source_path),
        "processed": _parquet_contract(processed_path),
        "ledger": {
            "name": ledger_path.name,
            "bytes": ledger_path.stat().st_size,
            "rows": diagnostics["rows"],
            "sha256": sha256_file(ledger_path),
        },
        "diagnostics": diagnostics,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "platform": platform.platform(),
        },
        "builder": {
            "name": Path(__file__).name,
            "sha256": sha256_file(Path(__file__)),
        },
        "licence_note": (
            "The ledger is derived from licensed CRSP data and must remain "
            "private.  Only aggregate diagnostics and cryptographic digests "
            "may be disclosed publicly."
        ),
    }

    temporary_manifest = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)

    print("REFERENCE LEDGER CREATED")
    print(f"  ledger:   {ledger_path}")
    print(f"  manifest: {manifest_path}")
    print(
        "  rows: "
        f"{diagnostics['rows']} "
        f"({diagnostics['stochastic_legacy']} stochastic, "
        f"{diagnostics['deterministic_zero']} deterministic zero)"
    )
    print(
        "  max adjusted-return reconstruction error: "
        f"{diagnostics['max_adjusted_return_error']:.3e}"
    )


def load_and_validate_ledger(ledger_path: Path) -> pd.DataFrame:
    """Load the private ledger and enforce its reference schema contract."""
    ledger = pd.read_parquet(ledger_path)
    required = {
        *KEY_COLUMNS,
        "DelRet_imputed",
        "DlyRet_raw",
        "DlyRet_adjusted",
        "imputation_method",
    }
    missing = required.difference(ledger.columns)
    if missing:
        raise AssertionError(f"ledger is missing columns: {sorted(missing)}")

    ledger["DlyCalDt"] = pd.to_datetime(ledger["DlyCalDt"])
    ledger["PERMNO"] = ledger["PERMNO"].astype("int64")
    if ledger.duplicated(KEY_COLUMNS).any():
        raise AssertionError("ledger contains duplicate (PERMNO, DlyCalDt) keys")
    if ledger["DelRet_imputed"].isna().any():
        raise AssertionError("ledger contains missing DelRet_imputed values")
    return ledger


def validate_ledger(
    *,
    source_path: Path,
    processed_path: Path,
    ledger_path: Path,
    manifest_path: Path | None,
) -> None:
    """Re-extract the historical ledger and compare it with the reference."""
    reference = load_and_validate_ledger(ledger_path)
    reconstructed, diagnostics = extract_historical_ledger(
        source_path,
        processed_path,
    )

    compare_columns = [
        "PERMNO",
        "DlyCalDt",
        "DelistingDt",
        "DelReasonType",
        "PrimaryExch",
        "delist_category",
        "imputation_method",
        "DelRet_imputed",
        "DlyRet_raw",
        "DlyRet_adjusted",
        "source_row_group",
        "source_row_offset",
    ]
    pd.testing.assert_frame_equal(
        reference[compare_columns].reset_index(drop=True),
        reconstructed[compare_columns].reset_index(drop=True),
        check_exact=True,
        check_dtype=True,
    )

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise AssertionError("manifest schema version mismatch")
        if manifest["ledger"]["sha256"] != sha256_file(ledger_path):
            raise AssertionError("ledger SHA-256 mismatch")
        if manifest["source"]["sha256"] != sha256_file(source_path):
            raise AssertionError("source SHA-256 mismatch")
        if manifest["processed"]["sha256"] != sha256_file(processed_path):
            raise AssertionError("processed SHA-256 mismatch")

    print("REFERENCE LEDGER VALIDATION PASS")
    print(f"  exact rows: {diagnostics['rows']}")
    print(
        "  composition: "
        f"{diagnostics['stochastic_legacy']} stochastic, "
        f"{diagnostics['deterministic_zero']} deterministic zero"
    )
    print(
        "  max adjusted-return reconstruction error: "
        f"{diagnostics['max_adjusted_return_error']:.3e}"
    )


def reconstruct_panel(
    *,
    source_path: Path,
    ledger_path: Path | None,
    output_path: Path,
    mode: str,
    base_seed: int,
) -> None:
    """Stream a reconstructed processed panel from the unprocessed source."""
    if mode not in {"reference", "seeded", "extend"}:
        raise ValueError("mode must be 'reference', 'seeded' or 'extend'")

    source_resolved = source_path.resolve()
    output_resolved = output_path.resolve()
    if source_resolved == output_resolved:
        raise ValueError("output path must differ from source path")

    if mode in {"reference", "extend"}:
        if ledger_path is None or not ledger_path.exists():
            raise FileNotFoundError(
                f"mode '{mode}' requires the private reference ledger"
            )
        ledger = load_and_validate_ledger(ledger_path)
    else:
        ledger = pd.DataFrame(columns=[*KEY_COLUMNS, "DelRet_imputed"])

    ledger_map = {
        (int(row.PERMNO), pd.Timestamp(row.DlyCalDt)): float(
            row.DelRet_imputed
        )
        for row in ledger.itertuples(index=False)
    }
    ledger_keys = set(ledger_map)
    matched_ledger_keys: set[tuple[int, pd.Timestamp]] = set()
    seeded_rows = 0

    source_file = pq.ParquetFile(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    writer: pq.ParquetWriter | None = None
    try:
        for row_group in range(source_file.metadata.num_row_groups):
            chunk = source_file.read_row_group(row_group).to_pandas()
            chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
            chunk["DelistingDt"] = pd.to_datetime(chunk["DelistingDt"])

            event_day = (
                chunk["DelistingDt"].notna()
                & (chunk["DlyCalDt"] == chunk["DelistingDt"])
            )
            chunk["delist_category"] = np.where(
                event_day,
                chunk["DelReasonType"].map(DELISTING_MAPPING),
                np.nan,
            )
            needs_imputation = event_day & chunk["DelRet"].isna()

            for row_index in chunk.index[needs_imputation]:
                permno = int(chunk.at[row_index, "PERMNO"])
                date = pd.Timestamp(chunk.at[row_index, "DlyCalDt"])
                key = (permno, date)

                if mode in {"reference", "extend"} and key in ledger_map:
                    value = ledger_map[key]
                    matched_ledger_keys.add(key)
                elif mode in {"seeded", "extend"}:
                    value = seeded_imputation(
                        permno=permno,
                        date=date,
                        reason=chunk.at[row_index, "DelReasonType"],
                        exchange=chunk.at[row_index, "PrimaryExch"],
                        base_seed=base_seed,
                    )
                    seeded_rows += 1
                else:
                    raise AssertionError(
                        "reference reconstruction encountered an event absent "
                        f"from the ledger: PERMNO={permno}, date={date.date()}"
                    )

                chunk.at[row_index, "DelRet"] = value

            adjusted = (
                (1.0 + chunk["DlyRet"].fillna(0.0))
                * (1.0 + chunk["DelRet"].fillna(0.0))
                - 1.0
            )
            both_missing = chunk["DlyRet"].isna() & chunk["DelRet"].isna()
            adjusted.loc[both_missing] = np.nan
            chunk.loc[event_day, "DlyRet"] = adjusted.loc[event_day]

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output,
                    table.schema,
                    compression="snappy",
                )
            writer.write_table(table)

            print(
                f"row group {row_group + 1}/"
                f"{source_file.metadata.num_row_groups}"
            )

        if writer is not None:
            writer.close()
            writer = None

        missing_historical = ledger_keys.difference(matched_ledger_keys)
        if mode == "reference" and missing_historical:
            preview = sorted(missing_historical)[:5]
            raise AssertionError(
                f"{len(missing_historical)} ledger keys were not used; "
                f"first keys: {preview}"
            )
        if mode == "reference" and seeded_rows:
            raise AssertionError("reference reconstruction used seeded draws")

        os.replace(temporary_output, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    print("PANEL RECONSTRUCTION COMPLETE")
    print(f"  mode: {mode}")
    print(f"  historical ledger rows used: {len(matched_ledger_keys)}")
    print(f"  seeded rows generated: {seeded_rows}")
    print(f"  output: {output_path}")
    print(f"  SHA-256: {sha256_file(output_path)}")


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-ledger",
        help="extract the exact historical reference realisation",
    )
    build.add_argument("--source", type=_path, default=DEFAULT_SOURCE)
    build.add_argument("--processed", type=_path, default=DEFAULT_PROCESSED)
    build.add_argument("--ledger", type=_path, default=DEFAULT_LEDGER)
    build.add_argument("--manifest", type=_path, default=DEFAULT_MANIFEST)

    validate = subparsers.add_parser(
        "validate-ledger",
        help="validate the ledger against the reference input and output",
    )
    validate.add_argument("--source", type=_path, default=DEFAULT_SOURCE)
    validate.add_argument("--processed", type=_path, default=DEFAULT_PROCESSED)
    validate.add_argument("--ledger", type=_path, default=DEFAULT_LEDGER)
    validate.add_argument("--manifest", type=_path, default=DEFAULT_MANIFEST)
    validate.add_argument(
        "--skip-manifest-hashes",
        action="store_true",
        help="skip the expensive full-file SHA-256 checks",
    )

    reconstruct = subparsers.add_parser(
        "reconstruct",
        help="stream a reference, seeded or extended processed panel",
    )
    reconstruct.add_argument("--source", type=_path, default=DEFAULT_SOURCE)
    reconstruct.add_argument("--ledger", type=_path, default=DEFAULT_LEDGER)
    reconstruct.add_argument("--output", type=_path, required=True)
    reconstruct.add_argument(
        "--mode",
        choices=["reference", "seeded", "extend"],
        default="reference",
    )
    reconstruct.add_argument("--base-seed", type=int, default=BASE_SEED)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "build-ledger":
        build_ledger(
            source_path=args.source,
            processed_path=args.processed,
            ledger_path=args.ledger,
            manifest_path=args.manifest,
        )
    elif args.command == "validate-ledger":
        validate_ledger(
            source_path=args.source,
            processed_path=args.processed,
            ledger_path=args.ledger,
            manifest_path=(
                None if args.skip_manifest_hashes else args.manifest
            ),
        )
    elif args.command == "reconstruct":
        reconstruct_panel(
            source_path=args.source,
            ledger_path=args.ledger,
            output_path=args.output,
            mode=args.mode,
            base_seed=args.base_seed,
        )
    else:
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
