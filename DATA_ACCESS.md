# Data access and redistribution policy

The empirical replication uses licensed market data that cannot be bundled
with the public source code. This document separates the public processing
contract from the restricted observations required to execute it.

## Data sources

The thesis uses:

- CRSP US Stock Database security, return, exchange, capitalization, and
  delisting information obtained through WRDS;
- CBOE VIX observations used for the implied-volatility controls.

Users are responsible for obtaining authorized access from the relevant data
providers and for complying with their terms of use.

## What the repository does not distribute

The public repository excludes:

- raw or processed CRSP and CBOE observations;
- PERMNO-level identifiers, returns, universe memberships, holdings, and
  portfolio weights;
- the private delisting-imputation ledger used by the reference reconstruction;
- signal matrices, estimation windows, holding-period factors, caches, and
  audit ledgers;
- `timing_test_inputs.npz` and host-specific runtime guards.

The `.gitignore` applies a deny-by-default policy to the complete `data/` tree
and to common binary dataset formats. Only explicitly listed processing source
files are eligible for publication.

## Upstream reconstruction

The raw-data hand-off expected by the public processing code is a daily CRSP
security panel already merged with the CRSP delisting event table:

```text
CRSP daily security file + CRSP delisting event file
    -> merge on PERMNO
    -> remove observations after the selected delisting date
    -> data/crsp_daily_delistings_unprocessed.parquet
```

The canonical delisting adjustment is documented in
[`data/RAW_DATA_PS/README.md`](data/RAW_DATA_PS/README.md). It supports:

- `reference`, which reproduces the thesis panel when the restricted reference
  ledger is available;
- `seeded`, which generates a new deterministic realization using a user-set
  seed and stable SHA-256 observation keys;
- `extend`, which preserves reference events and deterministically processes
  new observations.

The output of that stage is consumed by
`notebooks/00_build_common_stock_source.ipynb`, which applies the point-in-time
CRSP CIZ ordinary-common-stock eligibility rules and retains the columns used
by the downstream pipeline.

## Main local pipeline

The public notebooks generate the restricted local artifacts in sequence:

```text
delisting-adjusted CRSP daily panel
    -> filtered ordinary-common-stock source
    -> point-in-time large-cap and NYSE small--mid-cap panels
    -> multi-frequency signal files
    -> portfolio paths, ledgers, and aggregate publication outputs
```

The exact filenames and assertions are encoded in the notebooks and their
matching validation notebooks. Generated files belong under `data/`, `cache/`,
and `outputs/`; these directories remain untracked.

## Timing-test input

The long-running permutation experiment additionally expects:

```text
data/processed/nyse_big_caps_pit_daily.parquet
data/processed/nyse_small_caps_p20_p50_pit_daily.parquet
data/signals/V_uni.parquet
data/signals/V_small_uni.parquet
outputs/results/portfolio/audit/portfolio_monthly.parquet
```

Its `prepare_inputs.py` command creates a restricted, signed local bundle. See
[`reproducibility/portfolio_timing_tests/DATA_ACCESS.md`](reproducibility/portfolio_timing_tests/DATA_ACCESS.md)
for the exact command and reference identity checks.

## Public verification without licensed inputs

The repository may distribute final figures, LaTeX tables, cryptographic
digests, signed manifests, and aggregate permutation statistics that do not
contain security-level observations. These artifacts allow readers to inspect
the reported inference and provenance, but they do not replace licensed inputs
for a full end-to-end reconstruction.

No data file should be committed merely because it is derived rather than raw.
Any proposed data artifact must first be checked for identifiers, return paths,
memberships, holdings, weights, or other information covered by the underlying
license.
