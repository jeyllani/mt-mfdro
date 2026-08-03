# MT-MFDRO

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](environment.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI: mfdro](https://img.shields.io/pypi/v/mfdro.svg)](https://pypi.org/project/mfdro/)

Replication repository for the master thesis **Distributionally Robust
Optimization: Endogenous Calibration of the Wasserstein Ambiguity Radius**,
submitted to HEC Lausanne, University of Lausanne.

## Abstract

This thesis examines whether the timing of the Wasserstein ambiguity radius can
be driven by observable information and have economic value distinct from its
average level. It constructs a misalignment signal from daily, weekly, and
monthly return distributions, aggregated around a Wasserstein barycenter and
compared using Sliced-Wasserstein distances. The signal modulates the radius of
a $W_2$ ambiguity set in a linear-loss portfolio problem, evaluated through a
point-in-time walk-forward design over 1995–2025. In the main universe, the
net Sharpe ratio increases from 0.6892 to 0.7451, a gain of 0.0559. Neither the
paired HAC test nor the paired bootstrap establishes a general improvement at
the 5% level. Signal-ordering placebos reject the interchangeability of the
observed temporal ordering, and the main specification remains significant
after max-$T$ adjustment within both the four-cell and twelve-cell families.
A modulation based on realized volatility nevertheless achieves a slightly
higher point estimate, without a significant paired difference. The
contribution is therefore primarily methodological: the level and timing of
the ambiguity radius can be distinguished and tested, but the signal's
incremental economic value has not been established.

## Repository scope

This repository contains the thesis-specific empirical pipeline:

- reconstruction of the delisting-adjusted CRSP daily panel;
- point-in-time CRSP CIZ common-stock eligibility;
- construction of the large-cap and NYSE small--mid-cap universes;
- estimation and validation of the multi-frequency signal;
- static and dynamically modulated DRO portfolio backtests;
- statistical inference, robustness exercises, and publication figures;
- a signed and resumable bundle for the long-running permutation and max-$T$
  experiment.

The reusable signal implementation is maintained separately as the
[`mfdro`](https://pypi.org/project/mfdro/) Python package:

```bash
python -m pip install mfdro
```

The package provides a parameterized public API for constructing the signal.
It does not reproduce the CRSP universes, portfolio engine, or thesis-specific
inference reported here, and it is not required to execute this repository.

## Repository layout

```text
mt-mfdro/
├── README.md
├── DATA_ACCESS.md
├── CITATION.cff
├── LICENSE
├── environment.yml
├── data/RAW_DATA_PS/             # publishable processing code only
├── notebooks/                    # empirical pipeline, steps 00--05
├── src/                          # active shared inference code
├── tests/                        # validation notebooks, steps 00--04
└── reproducibility/
    └── portfolio_timing_tests/   # signed permutation/max-T experiment
```

Licensed data, caches, security-level audit ledgers, local logs, and temporary
artifacts are deliberately excluded from version control.

## Environment

Create the reference environment from the repository root:

```bash
conda env create -f environment.yml
conda activate mthesis
python -m ipykernel install --user --name mthesis --display-name "Python (mthesis)"
```

The portfolio notebooks use MOSEK through CVXPY. Installing the Python package
does not provide a MOSEK license; users must supply a valid local academic or
commercial license. The signed long-run experiment must not be executed with a
silent solver fallback.

## Restricted data

The empirical pipeline relies on licensed CRSP/WRDS data. These data and all
security-level derivatives are **not distributed** in this repository. Users
must obtain their own authorized access and reconstruct the required local
files.

The raw-data reconstruction contract is documented in
[`data/RAW_DATA_PS/README.md`](data/RAW_DATA_PS/README.md). The additional
restricted inputs required by the timing experiment are documented in
[`reproducibility/portfolio_timing_tests/DATA_ACCESS.md`](reproducibility/portfolio_timing_tests/DATA_ACCESS.md).

The public repository contains only processing code, schemas, cryptographic
digests, publication-safe figures and tables, and aggregate inferential
results. In particular, it does not contain CRSP returns, PERMNO-level
holdings, portfolio weight ledgers, or the private timing input bundle.

## Notebook execution order

Run the notebooks from the repository root and preserve the following order:

| Step | Empirical notebook | Validation notebook | Purpose |
|---:|---|---|---|
| 0 | `notebooks/00_build_common_stock_source.ipynb` | `tests/00_assert_common_stock_source.ipynb` | Apply the point-in-time CIZ ordinary-common-stock filter and reduce the source columns. |
| 1 | `notebooks/01_build_pit_big_small_caps.ipynb` | `tests/01_assert_pit_big_small_caps.ipynb` | Construct the monthly large-cap and NYSE small--mid-cap universes. |
| 2 | `notebooks/02_compute_pit_signals.ipynb` | `tests/02_validate_and_publish_pit_signals.ipynb` | Compute, checkpoint, and validate the signal specifications. |
| 3 | `notebooks/03_R_Signal.ipynb` | `tests/03_assert_signal_validation.ipynb` | Produce and independently validate the signal results. |
| 4 | `notebooks/04_R_Portfolio.ipynb` | `tests/04_assert_portfolio_validation.ipynb` | Run the portfolio analysis, inference, robustness exercises, and exports. |
| 5 | `notebooks/05_generate_methodology_appendix_figures.ipynb` | -- | Generate the data-free methodology and appendix illustrations. |

The delisting-adjusted source used before step 0 is reconstructed by
`data/RAW_DATA_PS/01_crsp+del_processing.ipynb`. Its reference mode requires a
private CRSP-derived ledger that cannot be redistributed. Its seeded mode
provides a deterministic SHA-256-based construction for authorized users who
do not possess that ledger.

Public copies of all notebooks are intentionally output-free. Execute the
matching validation notebook immediately after each empirical stage rather
than only at the end of the pipeline.

## Long-running timing experiment

The permutation and family-wise max-$T$ calculations are isolated under
[`reproducibility/portfolio_timing_tests/`](reproducibility/portfolio_timing_tests/).
That bundle provides:

- a signed experiment manifest;
- cross-platform environment guards;
- resumable and checkpointed batches;
- status and merge utilities;
- SHA-256 verification of code and aggregate reference results.

Its [dedicated README](reproducibility/portfolio_timing_tests/README.md)
contains the exact smoke-test, batch, monitoring, and merge commands. The
restricted `timing_test_inputs.npz` file is rebuilt locally and never committed.

## Validation philosophy

The validation notebooks check more than successful execution. They verify
point-in-time membership, schemas, signed manifests, input and engine digests,
portfolio accounting identities, regression guards, permutation coverage,
exceedance counts, and exact row-wise max-$T$ reconstruction.

Reference results must never be combined across different input, experiment,
or engine digests. A methodological or data change defines a new experiment.

## Citation

If this repository contributes to academic work, please cite:

> Bakari, Abdul Kadir Jeylani (2026). *Distributionally Robust Optimization:
> Endogenous Calibration of the Wasserstein Ambiguity Radius*. Master thesis,
> HEC Lausanne, University of Lausanne.

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## License and disclaimer

The source code is released under the [MIT License](LICENSE). This license does
not grant redistribution rights for CRSP/WRDS data, CBOE data, or any other
third-party dataset. The thesis manuscript and institutional marks are not
covered by the software license.

This repository is provided for research and reproducibility purposes. It is
not investment advice, and no warranty is made regarding investment or trading
outcomes.
