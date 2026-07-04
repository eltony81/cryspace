# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CrySpace is a control systems analysis and design library for Crystal (a shard), inspired by python-control. It builds on `num.cr` (a fork at eltony81/num.cr) for LAPACK/BLAS-backed linear algebra. System dependencies: LAPACK, BLAS/CBLAS, and optionally Apache Arrow GLib.

## Commands

```bash
shards install                      # install dependencies into lib/

crystal spec                        # run all specs
crystal spec spec/pid_spec.cr       # run one spec file
crystal spec spec/pid_spec.cr:42    # run the spec at a specific line

crystal build --release src/your_app.cr            # standard CPU build
crystal build -Darrow --release src/your_app.cr    # Arrow SIMD backend
crystal build -Dopencl --release src/your_app.cr   # OpenCL GPU backend

crystal run examples/01_statespace_basic.cr        # run an example
```

- `shard.override.yml` (gitignored) points `opencl` and `arrow` at sibling checkouts (`../opencl.cr`, `../arrow.cr`) for local development.
- `scratch/` holds one-off debug/tuning scripts; it is not part of the library.

## Architecture

Everything lives under the `CrySpace` module. `src/cryspace.cr` just requires `src/cryspace/*` and defines `VERSION`.

**Reopened-class file layout**: the two core classes are split across many files that each reopen the class to add one functional area. When adding a method, put it in the file matching its domain (or follow the pattern with a new file):

- `CrySpace::StateSpace` — core in `statespace.cr` (matrices A/B/C/D, `dt : Float64?` where `nil` = continuous-time, dimension validation); extended by `statespace_simulation.cr` (step/impulse/lsim/simulate), `statespace_analysis.cr` (poles, gramians, margins), `statespace_synthesis.cr` (lqr/lqe/place/h2syn/hinfsyn), `statespace_realization.cr` (canonical forms), `statespace_reduction.cr` (balred/minreal), `statespace_connections.cr` (series/parallel/feedback), `statespace_conversion.cr` (sample/to_continuous/ss2tf), `statespace_freq_analysis.cr` (bode/nyquist/nichols data).
- `CrySpace::TransferFunction` — core in `transferfunction.cr` (num/den polynomials, denominator normalized so den[0] = 1.0); extended by `transferfunction_arithmetic.cr` (+, *, feedback, minreal) and `transferfunction_freq.cr`.

Other components are one class/module per file: `KalmanFilter`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `LuenbergerObserver`, `PIDController` (anti-windup), `Solver` (Euler/RK4/RK45 ODE solvers), `Tuning` (ZN/Cohen-Coon), `Ident`/`IdentFit` (system identification), `Nonlinear`/`AdaptiveNonlinear` (describing functions, iLQR, MRAC), `Plot` (self-contained HTML/Chart.js dashboards).

**Backend dispatch via compile-time flags**: `{% if flag?(:arrow) %}` blocks (in `statespace.cr` and `statespace_simulation.cr`) define `AnyFloat64Tensor` as a union of CPU and ARROW tensor types and offload `simulate` to the Arrow backend when the caller passes Arrow-backed tensors. Code must compile both with and without `-Darrow`; keep Arrow-specific code inside these macro blocks.

**Conventions**: matrices are 2D `Float64Tensor` (num.cr) built with `.to_tensor`; scalars are read with `.value` (e.g. `m[0, 0].value`). Vectorized simulation results are `[time_steps x n]` tensors; manual-iteration APIs (`step_response`) return arrays of tensors.

## Releasing a version bump

The version string appears in three places that must stay in sync: `shard.yml`, `CrySpace::VERSION` in `src/cryspace.cr`, and the badge at the top of `README.md`. Bumps of the num.cr dependency update `shard.yml` (and README mentions) — see recent `chore:` commits for the pattern.
