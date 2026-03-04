# pyIHW Design Document

**Date**: 2026-03-04
**Status**: Approved

## Overview

Python reimplementation of Independent Hypothesis Weighting (IHW) from the R/Bioconductor package by Ignatiadis & Huber. Given p-values, an independent covariate, and a significance level alpha, IHW learns data-driven weights via cross-validation and linear programming to improve power in large-scale multiple testing while controlling FDR (or FWER).

## Scope Decisions

- **Grenander + LP path only** — the deprecated ECDF/MILP path from R is excluded
- **LP solver**: `scipy.optimize.linprog` (HiGHS backend) — no extra dependencies; the LP is small (~3*nbins variables, ~hundreds of constraints)
- **Grenander estimator**: custom NumPy implementation (~20 lines, upper convex hull on sorted ECDF points)
- **Adjustment types**: both BH (FDR) and Bonferroni (FWER)
- **Null proportion estimation**: included (Storey's weighted pi0 estimator, off by default)

## Architecture: Functional Pipeline

Stateless functions in dedicated modules. A frozen dataclass for the result. No stateful classes.

```
ihw(pvalues, covariates, alpha, ...) -> IHWResult
```

### Module Layout

```
src/pyihw/
├── __init__.py      # Re-exports: ihw, IHWResult, bh_threshold
├── _types.py        # IHWResult dataclass, type aliases
├── ihw.py           # ihw() entry point, _ihw_internal() k-fold loop
├── weighting.py     # ihw_convex(), thresholds_to_weights(), penalty functions
├── splitting.py     # groups_by_filter(), assign_folds()
└── utils.py         # grenander_estimator(), bh_threshold(), bh_adjust(),
                     #   weighted_storey_pi0(), weighted_pvalues()
```

## Module Details

### `_types.py`

```python
@dataclasses.dataclass(frozen=True)
class IHWResult:
    pvalues: NDArray[np.float64]
    adj_pvalues: NDArray[np.float64]
    weights: NDArray[np.float64]           # per-hypothesis
    weighted_pvalues: NDArray[np.float64]
    covariates: NDArray[np.float64]
    groups: NDArray[np.intp]               # bin assignment per hypothesis
    folds: NDArray[np.intp]                # fold assignment per hypothesis
    weight_matrix: NDArray[np.float64]     # (nbins, nfolds)
    alpha: float
    nbins: int
    nfolds: int
    regularization_terms: NDArray[np.float64]  # lambda chosen per fold
    m_groups: NDArray[np.intp]             # hypotheses per stratum
    penalty: str                           # "total_variation" | "uniform_deviation"
    covariate_type: str                    # "ordinal" | "nominal"
    adjustment_type: str                   # "bh" | "bonferroni"
```

Properties: `n_rejections`, `rejected_hypotheses` (boolean mask).

### `splitting.py`

- **`groups_by_filter(covariates, nbins, rng)`**: Rank covariate, divide into `nbins` equal-size groups. Ties broken randomly via `rng`. Returns 0-indexed integer array.
- **`assign_folds(n, nfolds, rng)`**: Random fold assignment via `rng.integers(0, nfolds, size=n)`.

### `utils.py`

- **`grenander_estimator(sorted_pvalues, m_total)`**: ECDF -> least concave majorant via upper convex hull. Returns knot x-coords, y-coords, and slopes.
- **`bh_threshold(pvalues, alpha, m_total=None)`**: BH rejection threshold — largest `p_(i) <= i/m * alpha`.
- **`bh_adjust(pvalues, m_total=None)`**: BH-adjusted p-values.
- **`weighted_storey_pi0(pvalues, weights, tau=0.5, m=None)`**: Weighted Storey pi0 estimator.
- **`weighted_pvalues(pvalues, weights)`**: `p / w` with guards for zero weights/p-values.

### `weighting.py`

- **`ihw_convex(split_sorted_pvalues, alpha, m_groups, m_groups_grenander, penalty, lambda_, adjustment_type)`**:
  1. Short-circuit to uniform weights if `lambda_ == 0`
  2. Clip p-values < 1e-20 to 0
  3. Grenander estimator per bin
  4. Build LP: Grenander constraints, FDR/FWER constraint, regularization (TV or UD with auxiliary variables)
  5. Objective: maximize `sum(m_g/m * nbins * y_g)`
  6. Solve via `scipy.optimize.linprog`
  7. Convert thresholds to weights

- **`thresholds_to_weights(thresholds, m_groups)`**: `w_g = t_g * m / sum(m_g * t_g)`, ensuring mean weight = 1.
- **`total_variation(weights)`** / **`uniform_deviation(weights)`**: For validation.

### `ihw.py`

- **`ihw(pvalues, covariates, alpha, ...)`**: Public entry point.
  - Parameters: `covariate_type="ordinal"`, `nbins="auto"`, `nfolds=5`, `nfolds_internal=5`, `nsplits_internal=1`, `lambdas="auto"`, `adjustment_type="bh"`, `null_proportion=False`, `null_proportion_level=0.5`, `folds=None`, `rng=None`
  - Validates inputs, bins covariate, selects penalty, auto-generates lambda grid, sorts p-values, calls `_ihw_internal`, reorders, returns `IHWResult`.

- **`_ihw_internal(sorted_groups, sorted_pvalues, alpha, lambdas, m_groups, ...)`**: K-fold loop.
  - For each fold: partition data, nested CV for lambda selection (recursive call), solve `ihw_convex`, optionally apply Storey pi0 adjustment.
  - Computes weighted p-values, applies BH/Bonferroni adjustment.

### `__init__.py`

Re-exports: `ihw`, `IHWResult`, `bh_threshold`.

## Testing Strategy

```
tests/
├── conftest.py          # wasserman_sim fixture, fixed rng fixtures
├── test_splitting.py    # group sizes, bin counts, determinism
├── test_utils.py        # grenander vs R output, bh_threshold, storey pi0
├── test_weighting.py    # weight constraint, lambda=0, TV/UD bounds
└── test_ihw.py          # e2e: more rejections than BH, weight budget,
                         #   single fold/bin edge cases, bonferroni, reference comparison
```

Reference comparisons against R `IHW` output stored as CSV in `tests/data/`.

## Dependencies

- **Runtime**: `numpy`, `scipy`
- **Dev**: `pytest`, `ruff`, `ty`
