# Public research-output bundle

This directory contains identifier-free research results used by the public
execution path of notebooks 03 and 04. It does not contain CRSP source files,
security-level returns, portfolio constituents, PERMNO identifiers, or
security-level weights.

The bundle supports verification and materialization of the reported thesis
outputs without access to the licensed data. It is not a replacement for the
licensed inputs and does not reconstruct the empirical pipeline from raw data.
The full execution path remains available to authorized CRSP users.

## Contents

- `signal/`: validated scalar signal paths and aggregate placebo, power, ROC,
  and Monte-Carlo diagnostics;
- `portfolio/`: aggregate monthly strategy paths, summary metrics, robustness
  results, and inferential outputs;
- `MANIFEST.json`: machine-readable file schemas, dimensions, provenance, and
  SHA-256 digests.

Every released file is also covered by `artifacts/SHA256SUMS`. Files should
never be mixed across different signal, portfolio-engine, or timing-experiment
digests.

## Data provenance and use

These files are research results derived from licensed CRSP/WRDS inputs. The
underlying licensed data are not redistributed. Users seeking to reconstruct
the results from security-level observations must obtain their own authorized
access and follow `DATA_ACCESS.md`.

Source attribution: CRSP®, Center for Research in Security Prices. Used with
permission. All rights reserved.
