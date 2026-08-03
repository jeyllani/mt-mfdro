# CRSP daily delisting panel

This directory contains the canonical, memory-bounded construction of the
delisting-adjusted CRSP daily panel. The human-facing entry point is
`01_crsp+del_processing.ipynb`; the implementation is
`src/delisting_panel_builder.py`.

For the complete empirical workflow, environment, and publication policy, see
the [repository README](../../README.md).

## Input contract

The expected input, `data/crsp_daily_delistings_unprocessed.parquet`, is a
daily CRSP security panel already merged on `PERMNO` with the CRSP delisting
event table (`DEL`):

```text
CRSP.csv + DEL.csv
    -> daily merge and removal of post-delisting observations
    -> canonical delisting columns
    -> crsp_daily_delistings_unprocessed.parquet
```

It contains one security-date observation per row, together with
`DelistingDt`, `DelRet`, `DelReasonType`, `DelActionType`, and
`DelStatusType`. Observations strictly subsequent to the selected delisting
date have already been removed. Missing event-day delisting returns have not
yet been imputed and `DlyRet` has not yet been compounded with the terminal
return.

The reference input contains:

- 70,190,582 daily rows;
- 21,363 unique delisting records;
- 21,372 event rows;
- 1,244 missing delisting-return imputations.

## Construction modes

### `reference` — reproduce the thesis panel

```text
crsp_daily_delistings_unprocessed.parquet
    + private/delret_reference_ledger_v1.parquet
    -> exact reference event-day DelRet values
    -> exact adjusted-return reconstruction
```

This is the default mode. It generates no random value and fails if any
reference key is missing or unused.

### `seeded` — generate an alternative reproducible realisation

All missing event-day delisting returns are generated from a user-selected
base seed. A per-observation seed is derived from SHA-256 of
`PERMNO_YYYY-MM-DD`, making the result independent of the Python process,
machine, row order, and batch partition. Reusing the same base seed reproduces
the same logical panel. This mode requires only the merged unprocessed input;
it does not require the private reference ledger or the historical processed
panel.

### `extend` — preserve history and process new events

Reference ledger values are retained for known historical observations.
Events absent from the ledger are generated with the same SHA-256 seeded rule.

## Commands

Run from the project root in the `mthesis` environment:

```bash
python data/RAW_DATA_PS/src/delisting_panel_builder.py validate-ledger

python data/RAW_DATA_PS/src/delisting_panel_builder.py reconstruct \
  --mode reference \
  --output data/crsp_daily_shumway_delisting_reference_rebuilt.parquet

python data/RAW_DATA_PS/src/delisting_panel_builder.py reconstruct \
  --mode seeded \
  --base-seed 123 \
  --output data/crsp_daily_shumway_delisting_seeded_seed123.parquet

python data/RAW_DATA_PS/src/delisting_panel_builder.py reconstruct \
  --mode extend \
  --base-seed 42 \
  --output data/crsp_daily_shumway_delisting_extended_seed42.parquet
```

The reference ledger and manifest are CRSP-derived, remain private, and are
excluded from public version control. The manifest records the SHA-256 digests
of the unprocessed input, reference processed panel, ledger, and builder.
The notebook keeps full 4 GB reconstruction disabled by default.

## Historical provenance

The recovered historical RunPod producer, an unused Polars prototype, and the
obsolete standalone column-projection notebook are archived under
`tmp/RAW_DATA_PS_legacy/` and are not part of the active path.
The historical producer used a process-dependent Python hash to derive
per-observation seeds. The exact realised values used by the thesis are now
preserved by the private reference ledger; alternative generations use the
stable SHA-256 rule.

The active downstream hand-off is
`notebooks/00_build_common_stock_source.ipynb`. It reads the canonical full
processed panel, applies the point-in-time CRSP CIZ common-stock eligibility
rules, and writes the reduced source consumed by the universe builder.
