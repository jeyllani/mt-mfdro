# Portfolio timing tests

Reproducibility bundle for the three-level portfolio-timing inference reported
in my thesis. The experiment uses the corrected ordinary-common-stock
universes and the direct Wasserstein SOCP

\[
-\widehat{\mu}^{\top}w+\varepsilon\lVert w\rVert_2.
\]

The reference experiment is identified by digest
`88e42643b59e1a736ba20a8a722159bcc13724fb62b954967f68c2648cfbb2e8`.
Inputs or results from another digest must never be mixed with this experiment.

For the upstream notebook pipeline, repository scope, and licensing policy,
see the [repository README](../../README.md).

## Scope

This bundle is the standalone implementation of the shared timing-inference
panel loaded by Sections 6.2.4 and 6.2.8.1 of
[`04_R_Portfolio.ipynb`](../../notebooks/04_R_Portfolio.ipynb). It does not
reproduce only the final max-$T_{12}$ statistic: one signed and resumable run
computes all three pre-declared inferential levels from $B=1{,}000$ complete
portfolio reconstructions:

1. **Timing placebo (Section 6.2.4; Figure P_04 and Table P_02).** The observed
   net Sharpe of the dynamic large-cap portfolio is compared with the Sharpe
   obtained after simple monthly, six-month-block, and twelve-month-block
   rearrangements of the modulation path. Each rearranged signal is passed
   through the complete portfolio backtest; asset returns are never permuted.
2. **Primary max-$T_4$ family (Section 6.2.8.1; Figures P_08--P_09 and Tables
   P_06--P_07).** Each simple monthly draw produces four dynamic-minus-static
   $\Delta\mathrm{SR}$ statistics: full sample and ex-dot-com for the large-cap
   (`L_all`, `L_exdc`) and NYSE small--mid-cap (`S_all`, `S_exdc`) universes.
   Their row-wise maximum defines the multiplicity-adjusted null distribution.
3. **Extended max-$T_{12}$ family (Section 6.2.8.1; Table P_08).** The same
   monthly draws are expanded to twelve pre-declared cells by adding six
   non-overlapping universe--decade cells and two large-cap dispersion-state
   cells. Taking the maximum over this broader family gives a deliberately more
   conservative diagnostic.

Within each simple monthly draw, one permutation is shared by the large- and
small-cap modulation paths. This preserves cross-universe and nested-sample
dependence before the family maximum is computed. The simple permutation does
not preserve the signal's serial dependence; the six- and twelve-month block
designs address that issue only as separate level-1 robustness nulls and are
not used for the max-$T$ families. Marginal and family-wise upper-tail
$p$-values use the finite-sample correction $(r+1)/(B+1)$.

## Public and restricted artifacts

The repository contains the code, the signed reference manifest, and the
aggregate reference results. It deliberately excludes
`data/timing_test_inputs.npz`. That input bundle contains CRSP-derived security
identifiers, estimation returns, and holding-period factors and therefore
cannot be redistributed under the CRSP/WRDS data license.

Licensed users can reconstruct the private input bundle locally with
`prepare_inputs.py`. The expected SHA-256 of the input used for the published
experiment is recorded in `reference/INPUT_SHA256` and in the signed manifest.
See `DATA_ACCESS.md` for the required upstream files.

The files under `reference/results/` contain only draw-level or aggregate test
statistics; they contain no security-level returns or identifiers. They are
provided to make the reported inference auditable without redistributing CRSP
data.

## Directory layout

```text
portfolio_timing_tests/
├── README.md
├── DATA_ACCESS.md
├── environment.yml
├── SHA256SUMS_CODE
├── timing_core.py
├── prepare_inputs.py
├── validate_environment.py
├── run_batch.py
├── merge_results.py
├── status.py
└── reference/
    ├── INPUT_SHA256
    ├── timing_test_manifest.json
    └── results/
        ├── SHA256SUMS_RESULTS
        ├── timing_tests_B1000_88e42643b59e.parquet
        ├── level1_placebo_B1000_88e42643b59e.parquet
        ├── levels2_3_inference_B1000_88e42643b59e.parquet
        └── timing_tests_summary.json
```

The untracked `data/`, `logs/`, and `results/` directories are created during a
local reconstruction or a new run.

## Environment

Create the environment and activate it from the repository root:

```bash
conda env create -f reproducibility/portfolio_timing_tests/environment.yml
conda activate mthesis
cd reproducibility/portfolio_timing_tests
```

The reference VPS run used Python 3.12.13, NumPy 2.4.4, pandas 2.3.3,
CVXPY 1.8.2, and MOSEK 11.1.10. A valid MOSEK license is required.

## Verify the published files

From `reproducibility/portfolio_timing_tests`:

```bash
shasum -a 256 -c SHA256SUMS_CODE
cd reference/results
shasum -a 256 -c SHA256SUMS_RESULTS
cd ../..
```

## Reconstruct the restricted input bundle

This step requires licensed local CRSP-derived inputs and the upstream outputs
listed in `DATA_ACCESS.md`:

```bash
python -u prepare_inputs.py --project-root ../.. --threads 1
```

The command creates `data/timing_test_inputs.npz`,
`data/timing_test_manifest.json`, and a private-run `SHA256SUMS`. Confirm the
input identity with:

```bash
shasum -a 256 data/timing_test_inputs.npz
cat reference/INPUT_SHA256
```

An exact match reproduces the published input bundle. A mismatch defines a new
experiment and must not be combined with the reference results.

## Mandatory cross-platform numerical guard

Before running any null draw on a new host, reconstruct the observed static and
dynamic paths:

```bash
python -u validate_environment.py \
  --threads 1 \
  --return-tolerance 2e-6 \
  --turnover-tolerance 5e-5 \
  --cell-tolerance 2e-5 \
  2>&1 | tee logs/environment_guard.log
```

Do not start a batch unless the command ends with `ENVIRONMENT GUARD PASS`.
The input-preparation cross-contract and this cross-host runtime guard serve
different purposes: the former checks agreement with the producing notebook,
whereas the latter allows only the observed numerical differences recorded for
the Linux/aarch64 VPS environment. These tolerances do not alter any portfolio,
test statistic, or p-value.

## Smoke test

```bash
python -u run_batch.py \
  --start 0 --stop 2 \
  --checkpoint-every 1 \
  --progress-every 1 \
  --threads 1

python status.py
```

Each draw solves four complete paths: large/simple and small/simple with the
same monthly ordering, followed by large/block6 and large/block12.

## Full resumable run

```bash
nohup python -u run_batch.py \
  --start 2 --stop 1000 \
  --checkpoint-every 5 \
  --progress-every 5 \
  --threads 1 \
  > logs/nohup_00002_01000.out 2>&1 &

echo $! > logs/timing_tests.pid
```

Monitor the run with:

```bash
python status.py
```

or:

```bash
tail -f logs/batch_00002_01000.log
```

The process is crash-safe and resumes compatible atomic shards. Never combine
shards carrying different experiment or runtime digests.

## Merge and inference

When `status.py` reports `1000/1000`:

```bash
python -u merge_results.py 2>&1 | tee logs/merge_results.log
```

The command validates every draw and writes:

- `results/timing_tests_B1000_<digest>.parquet`;
- `results/level1_placebo_B1000_<digest>.parquet`;
- `results/levels2_3_inference_B1000_<digest>.parquet`;
- `results/timing_tests_summary.json`.

The level-2 and level-3 headline values are the `S_exdc` net rows. The code
asserts that the nested-family correction satisfies
`p_maxT12 >= p_maxT4`.

## Reproduction rules

- Never edit a generated manifest or NPZ in place.
- Never merge shards from different experiment or runtime digests.
- Never bypass the environment guard or enable a silent solver fallback.
- A methodological or input change requires a newly generated manifest and
  experiment digest.
