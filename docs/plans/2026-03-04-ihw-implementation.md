# pyIHW Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Independent Hypothesis Weighting (IHW) in Python — a multiple testing procedure that learns data-driven weights from an independent covariate to improve power while controlling FDR/FWER.

**Architecture:** Functional pipeline with stateless functions in dedicated modules (`splitting.py`, `utils.py`, `weighting.py`, `ihw.py`) and a frozen `IHWResult` dataclass. No stateful classes. See `docs/plans/2026-03-04-ihw-design.md` for full design.

**Tech Stack:** Python 3.13, NumPy, SciPy (`scipy.optimize.linprog`), pytest, ruff, ty, uv.

---

## Prerequisites

Before starting, add missing dependencies:

```bash
uv add scipy
uv add --dev pytest
uv sync
```

Create the test directories:

```bash
mkdir -p tests/data
touch tests/__init__.py
```

Commit:

```bash
git add pyproject.toml uv.lock tests/__init__.py
git commit -m "Add scipy and pytest dependencies, create tests directory"
```

---

### Task 1: `_types.py` — IHWResult dataclass

**Files:**
- Create: `src/pyihw/_types.py`
- Test: `tests/test_types.py`

**Context:** This is the result container returned by `ihw()`. Frozen dataclass with NumPy arrays for per-hypothesis data and scalar metadata. Two computed properties: `rejected_hypotheses` (boolean mask) and `n_rejections` (count).

**Step 1: Write the failing test**

Create `tests/test_types.py`:

```python
from __future__ import annotations

import numpy as np

from pyihw._types import IHWResult


def test_ihw_result_rejected_hypotheses() -> None:
    result = IHWResult(
        pvalues=np.array([0.01, 0.5, 0.001]),
        adj_pvalues=np.array([0.05, 0.5, 0.01]),
        weights=np.array([1.5, 0.5, 1.0]),
        weighted_pvalues=np.array([0.0067, 1.0, 0.001]),
        covariates=np.array([10.0, 2.0, 8.0]),
        groups=np.array([0, 1, 0]),
        folds=np.array([0, 1, 0]),
        weight_matrix=np.array([[1.5, 1.0], [0.5, 1.0]]),
        alpha=0.1,
        nbins=2,
        nfolds=2,
        regularization_terms=np.array([1.0, 1.0]),
        m_groups=np.array([2, 1]),
        penalty="total_variation",
        covariate_type="ordinal",
        adjustment_type="bh",
    )
    expected_mask = np.array([True, False, True])
    np.testing.assert_array_equal(result.rejected_hypotheses, expected_mask)
    assert result.n_rejections == 2


def test_ihw_result_is_frozen() -> None:
    result = IHWResult(
        pvalues=np.array([0.01]),
        adj_pvalues=np.array([0.05]),
        weights=np.array([1.0]),
        weighted_pvalues=np.array([0.01]),
        covariates=np.array([1.0]),
        groups=np.array([0]),
        folds=np.array([0]),
        weight_matrix=np.array([[1.0]]),
        alpha=0.1,
        nbins=1,
        nfolds=1,
        regularization_terms=np.array([0.0]),
        m_groups=np.array([1]),
        penalty="total_variation",
        covariate_type="ordinal",
        adjustment_type="bh",
    )
    try:
        result.alpha = 0.2  # type: ignore[misc]
        raise AssertionError("Should not be able to mutate frozen dataclass")
    except AttributeError:
        pass
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyihw._types'`

**Step 3: Write the implementation**

Create `src/pyihw/_types.py`:

```python
from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True)
class IHWResult:
    """Result container for Independent Hypothesis Weighting.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Original input p-values, in the original input order.
    adj_pvalues : NDArray[np.float64]
        Adjusted p-values after the weighted BH (or Bonferroni) procedure.
    weights : NDArray[np.float64]
        Per-hypothesis weights learned by IHW.
    weighted_pvalues : NDArray[np.float64]
        P-values divided by their weights (``pvalues / weights``).
    covariates : NDArray[np.float64]
        Original input covariates, in the original input order.
    groups : NDArray[np.intp]
        Bin (stratum) assignment for each hypothesis, 0-indexed.
    folds : NDArray[np.intp]
        Fold assignment for each hypothesis, 0-indexed.
    weight_matrix : NDArray[np.float64]
        Weight per (bin, fold) combination, shape ``(nbins, nfolds)``.
    alpha : float
        Nominal significance level.
    nbins : int
        Number of covariate bins (strata).
    nfolds : int
        Number of cross-validation folds.
    regularization_terms : NDArray[np.float64]
        Regularization parameter (lambda) chosen for each fold.
    m_groups : NDArray[np.intp]
        Number of hypotheses in each stratum.
    penalty : str
        Regularization penalty type: ``"total_variation"`` or
        ``"uniform_deviation"``.
    covariate_type : str
        Whether the covariate is ``"ordinal"`` or ``"nominal"``.
    adjustment_type : str
        Multiple testing adjustment: ``"bh"`` or ``"bonferroni"``.
    """

    pvalues: NDArray[np.float64]
    adj_pvalues: NDArray[np.float64]
    weights: NDArray[np.float64]
    weighted_pvalues: NDArray[np.float64]
    covariates: NDArray[np.float64]
    groups: NDArray[np.intp]
    folds: NDArray[np.intp]
    weight_matrix: NDArray[np.float64]
    alpha: float
    nbins: int
    nfolds: int
    regularization_terms: NDArray[np.float64]
    m_groups: NDArray[np.intp]
    penalty: str
    covariate_type: str
    adjustment_type: str

    @property
    def rejected_hypotheses(self) -> NDArray[np.bool_]:
        """Boolean mask of hypotheses rejected at level ``alpha``."""
        return self.adj_pvalues <= self.alpha

    @property
    def n_rejections(self) -> int:
        """Total number of rejected hypotheses."""
        return int(np.sum(self.rejected_hypotheses))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_types.py -v`
Expected: 2 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/_types.py tests/test_types.py
uv run ruff check src/pyihw/_types.py tests/test_types.py
git add src/pyihw/_types.py tests/test_types.py
git commit -m "Add IHWResult frozen dataclass with rejected_hypotheses property"
```

---

### Task 2: `splitting.py` — Binning and fold assignment

**Files:**
- Create: `src/pyihw/splitting.py`
- Test: `tests/test_splitting.py`

**Context:** Two functions. `groups_by_filter` ranks a numeric covariate and divides hypotheses into `nbins` equal-size groups (max size difference of 1). `assign_folds` randomly assigns hypotheses to folds. Both take a `numpy.random.Generator` for reproducibility.

**Step 1: Write the failing tests**

Create `tests/test_splitting.py`:

```python
from __future__ import annotations

import numpy as np

from pyihw.splitting import assign_folds, groups_by_filter


def test_groups_by_filter_equal_sizes() -> None:
    rng = np.random.default_rng(42)
    covariates = rng.uniform(size=1000)
    groups = groups_by_filter(covariates, nbins=7, rng=rng)
    counts = np.bincount(groups)
    assert counts.max() - counts.min() <= 1


def test_groups_by_filter_correct_nbins() -> None:
    rng = np.random.default_rng(42)
    covariates = rng.uniform(size=100)
    groups = groups_by_filter(covariates, nbins=10, rng=rng)
    assert len(np.unique(groups)) == 10


def test_groups_by_filter_preserves_order() -> None:
    """Higher covariate values should map to higher group indices."""
    rng = np.random.default_rng(42)
    covariates = np.arange(100, dtype=np.float64)
    groups = groups_by_filter(covariates, nbins=5, rng=rng)
    # first 20 elements should be in group 0, last 20 in group 4
    assert np.all(groups[:20] == 0)
    assert np.all(groups[80:] == 4)


def test_groups_by_filter_deterministic() -> None:
    covariates = np.array([0.5, 0.5, 0.5, 0.1, 0.9])
    g1 = groups_by_filter(covariates, nbins=2, rng=np.random.default_rng(99))
    g2 = groups_by_filter(covariates, nbins=2, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(g1, g2)


def test_assign_folds_correct_shape() -> None:
    rng = np.random.default_rng(42)
    folds = assign_folds(1000, nfolds=5, rng=rng)
    assert folds.shape == (1000,)
    assert set(np.unique(folds)).issubset(set(range(5)))


def test_assign_folds_deterministic() -> None:
    f1 = assign_folds(100, nfolds=3, rng=np.random.default_rng(7))
    f2 = assign_folds(100, nfolds=3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(f1, f2)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splitting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyihw.splitting'`

**Step 3: Write the implementation**

Create `src/pyihw/splitting.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def groups_by_filter(
    covariates: NDArray[np.float64],
    nbins: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Stratify hypotheses into equal-size bins by covariate rank.

    Parameters
    ----------
    covariates : NDArray[np.float64]
        Numeric covariate values, one per hypothesis.
    nbins : int
        Number of bins to create.
    rng : numpy.random.Generator
        Random number generator for breaking ties.

    Returns
    -------
    NDArray[np.intp]
        0-indexed bin assignment for each hypothesis. Bins are ordered
        so that lower covariate values map to lower bin indices.

    Notes
    -----
    Ties in the covariate are broken randomly using *rng*. The resulting
    bins differ in size by at most 1.
    """
    n = len(covariates)
    # Break ties randomly: add tiny jitter based on a random permutation
    jitter = rng.permutation(n).astype(np.float64) / (n + 1)
    # Rank from 0..n-1 using argsort of (covariate, jitter)
    order = np.lexsort((jitter, covariates))
    ranks = np.empty(n, dtype=np.intp)
    ranks[order] = np.arange(n)
    # Map ranks to bins: bin = rank * nbins // n
    return (ranks * nbins) // n


def assign_folds(
    n: int,
    nfolds: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Randomly assign hypotheses to cross-validation folds.

    Parameters
    ----------
    n : int
        Number of hypotheses.
    nfolds : int
        Number of folds.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    NDArray[np.intp]
        0-indexed fold assignment for each hypothesis.
    """
    return rng.integers(0, nfolds, size=n)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_splitting.py -v`
Expected: 6 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/splitting.py tests/test_splitting.py
uv run ruff check src/pyihw/splitting.py tests/test_splitting.py
git add src/pyihw/splitting.py tests/test_splitting.py
git commit -m "Add groups_by_filter and assign_folds for covariate binning"
```

---

### Task 3: `utils.py` — BH threshold and adjustment

**Files:**
- Create: `src/pyihw/utils.py`
- Test: `tests/test_utils.py`

**Context:** Start with the simpler statistical utilities before the Grenander estimator. `bh_threshold` finds the BH step-up cutoff. `bh_adjust` computes adjusted p-values. `safe_divide` handles p/w with zero guards. These are all needed before we can build higher-level functions.

**Step 1: Write the failing tests**

Create `tests/test_utils.py`:

```python
from __future__ import annotations

import numpy as np

from pyihw.utils import bh_adjust, bh_threshold, safe_divide


class TestBhThreshold:
    def test_known_values(self) -> None:
        pvalues = np.array([0.001, 0.01, 0.03, 0.04, 0.5, 0.8])
        # BH at alpha=0.1: sorted p_i vs i/6*0.1 = [0.0167, 0.0333, 0.05, 0.0667, 0.0833, 0.1]
        # 0.001 <= 0.0167 yes, 0.01 <= 0.0333 yes, 0.03 <= 0.05 yes, 0.04 <= 0.0667 yes
        # 0.5 <= 0.0833 no => threshold is 0.04
        assert bh_threshold(pvalues, alpha=0.1) == 0.04

    def test_no_rejections(self) -> None:
        pvalues = np.array([0.5, 0.6, 0.7])
        assert bh_threshold(pvalues, alpha=0.01) == 0.0

    def test_all_rejected(self) -> None:
        pvalues = np.array([0.001, 0.002, 0.003])
        assert bh_threshold(pvalues, alpha=0.5) == 0.003

    def test_with_mtotal_larger(self) -> None:
        # If m_total > len(pvalues), thresholds are more conservative
        pvalues = np.array([0.001, 0.01, 0.04])
        t1 = bh_threshold(pvalues, alpha=0.1)
        t2 = bh_threshold(pvalues, alpha=0.1, m_total=100)
        assert t2 <= t1


class TestBhAdjust:
    def test_equivalence_with_threshold(self) -> None:
        rng = np.random.default_rng(1)
        pvalues = np.concatenate([rng.uniform(size=1000), rng.beta(0.5, 7, size=200)])
        alpha = 0.1
        adj = bh_adjust(pvalues)
        t = bh_threshold(pvalues, alpha)
        # Rejections via adjusted p-values should match rejections via threshold
        np.testing.assert_array_equal(adj <= alpha, pvalues <= t)

    def test_adjusted_pvalues_are_monotone_on_sorted(self) -> None:
        pvalues = np.array([0.001, 0.01, 0.05, 0.5])
        adj = bh_adjust(pvalues)
        sorted_adj = adj[np.argsort(pvalues)]
        assert np.all(np.diff(sorted_adj) >= 0)

    def test_adjusted_pvalues_clipped_to_one(self) -> None:
        pvalues = np.array([0.9, 0.95, 0.99])
        adj = bh_adjust(pvalues)
        assert np.all(adj <= 1.0)

    def test_with_mtotal(self) -> None:
        pvalues = np.array([0.01, 0.02])
        adj1 = bh_adjust(pvalues)
        adj2 = bh_adjust(pvalues, m_total=100)
        # More tests => larger adjusted p-values
        assert np.all(adj2 >= adj1)


class TestSafeDivide:
    def test_normal_division(self) -> None:
        np.testing.assert_allclose(safe_divide(np.array([0.1]), np.array([2.0])), [0.05])

    def test_zero_numerator(self) -> None:
        # 0/w = 0 regardless of w
        np.testing.assert_array_equal(safe_divide(np.array([0.0]), np.array([0.0])), [0.0])

    def test_zero_denominator_nonzero_numerator(self) -> None:
        # p/0 = 1 (conservative)
        np.testing.assert_array_equal(safe_divide(np.array([0.5]), np.array([0.0])), [1.0])

    def test_result_clipped_to_one(self) -> None:
        # 0.8 / 0.5 = 1.6 -> clipped to 1.0
        np.testing.assert_array_equal(safe_divide(np.array([0.8]), np.array([0.5])), [1.0])
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyihw.utils'`

**Step 3: Write the implementation**

Create `src/pyihw/utils.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bh_threshold(
    pvalues: NDArray[np.float64],
    alpha: float,
    m_total: int | None = None,
) -> float:
    """Compute the Benjamini-Hochberg step-up rejection threshold.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Vector of p-values.
    alpha : float
        Significance level.
    m_total : int, optional
        Total number of tests. Defaults to ``len(pvalues)``. Set this
        to a larger value when only a subset of p-values is provided.

    Returns
    -------
    float
        The BH threshold *t* such that hypotheses with ``p <= t`` are
        rejected. Returns 0.0 if no hypothesis is rejected.
    """
    m = len(pvalues)
    if m_total is None:
        m_total = m
    sorted_p = np.sort(pvalues)
    # BH criterion: p_(i) <= i / m_total * alpha
    thresholds = np.arange(1, m + 1) / m_total * alpha
    rejected = np.where(sorted_p <= thresholds)[0]
    if len(rejected) == 0:
        return 0.0
    return float(sorted_p[rejected[-1]])


def bh_adjust(
    pvalues: NDArray[np.float64],
    m_total: int | None = None,
) -> NDArray[np.float64]:
    """Compute Benjamini-Hochberg adjusted p-values.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Vector of unadjusted p-values.
    m_total : int, optional
        Total number of tests. Defaults to ``len(pvalues)``.

    Returns
    -------
    NDArray[np.float64]
        BH-adjusted p-values, same length as input.

    Notes
    -----
    Equivalent to R's ``p.adjust(p, method="BH", n=m_total)``. Adjusted
    p-values are clipped to [0, 1] and are monotone non-decreasing when
    sorted by the original p-values.
    """
    m = len(pvalues)
    if m_total is None:
        m_total = m
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    # adj_p[i] = min(p_(i) * m_total / i, adj_p[i+1])  (cumulative min from right)
    adjusted = sorted_p * m_total / np.arange(1, m + 1)
    # Enforce monotonicity: cumulative min from the right
    np.minimum.accumulate(adjusted[::-1], out=adjusted[::-1])
    adjusted = np.clip(adjusted, 0.0, 1.0)
    # Restore original order
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def safe_divide(
    pvalues: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute weighted p-values ``pvalues / weights`` with safety guards.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Numerator p-values.
    weights : NDArray[np.float64]
        Denominator weights.

    Returns
    -------
    NDArray[np.float64]
        ``pvalues / weights``, where: zero numerators always yield 0,
        zero denominators with nonzero numerators yield 1, and results
        are clipped to [0, 1].

    Notes
    -----
    Mirrors the ``mydiv`` helper in the R IHW package.
    """
    result = np.where(
        pvalues == 0.0,
        0.0,
        np.where(weights == 0.0, 1.0, np.minimum(pvalues / np.where(weights == 0.0, 1.0, weights), 1.0)),
    )
    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_utils.py -v`
Expected: 11 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/utils.py tests/test_utils.py
uv run ruff check src/pyihw/utils.py tests/test_utils.py
git add src/pyihw/utils.py tests/test_utils.py
git commit -m "Add BH threshold, BH adjust, and safe_divide utilities"
```

---

### Task 4: `utils.py` — Grenander estimator

**Files:**
- Modify: `src/pyihw/utils.py` (add function)
- Modify: `tests/test_utils.py` (add tests)

**Context:** The Grenander estimator computes the least concave majorant (LCM) of the ECDF of p-values within a stratum. This is used inside the LP to linearize the cumulative distribution. The algorithm: (1) compute ECDF at unique p-value points, (2) compute upper convex hull of those (x, y) points, (3) return the knot positions and slopes of the hull segments. The slopes are the piecewise-constant density estimates.

Reference: R's `fdrtool::gcmlcm(x, y, type="lcm")`.

**Step 1: Write the failing tests**

Add to `tests/test_utils.py`:

```python
from pyihw.utils import grenander_estimator


class TestGrenanderEstimator:
    def test_uniform_pvalues(self) -> None:
        """For uniform p-values, LCM of ECDF is close to the identity line."""
        rng = np.random.default_rng(42)
        pvalues = np.sort(rng.uniform(size=10000))
        result = grenander_estimator(pvalues, m_total=10000)
        # Slopes should all be close to 1.0 (uniform density)
        assert np.all(result.slopes >= 0.9)
        assert np.all(result.slopes <= 1.1)

    def test_knots_are_sorted(self) -> None:
        rng = np.random.default_rng(42)
        pvalues = np.sort(rng.beta(0.5, 7, size=500))
        result = grenander_estimator(pvalues, m_total=500)
        assert np.all(np.diff(result.x_knots) > 0)
        assert np.all(np.diff(result.y_knots) >= 0)

    def test_slopes_are_nonincreasing(self) -> None:
        """LCM slopes must be non-increasing (concavity)."""
        rng = np.random.default_rng(42)
        pvalues = np.sort(rng.beta(0.5, 7, size=500))
        result = grenander_estimator(pvalues, m_total=500)
        assert np.all(np.diff(result.slopes) <= 1e-12)

    def test_m_total_larger_than_sample(self) -> None:
        """When m_total > len(pvalues), ECDF values are scaled down."""
        pvalues = np.sort(np.array([0.01, 0.02, 0.05, 0.1, 0.3]))
        result = grenander_estimator(pvalues, m_total=100)
        # ECDF at max p-value should be 5/100 = 0.05, not 1.0
        assert result.y_knots[-1] <= 0.06

    def test_single_pvalue(self) -> None:
        result = grenander_estimator(np.array([0.5]), m_total=1)
        assert len(result.slopes) >= 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py::TestGrenanderEstimator -v`
Expected: FAIL — `ImportError: cannot import name 'grenander_estimator'`

**Step 3: Write the implementation**

Add to `src/pyihw/utils.py`:

```python
import dataclasses

@dataclasses.dataclass(frozen=True)
class GrenanderResult:
    """Result of the Grenander estimator (least concave majorant of ECDF).

    Parameters
    ----------
    x_knots : NDArray[np.float64]
        X-coordinates of the LCM knot points (left endpoints of segments).
    y_knots : NDArray[np.float64]
        Y-coordinates of the LCM knot points.
    slopes : NDArray[np.float64]
        Slope of each LCM segment. Length equals ``len(x_knots)``.
        Slopes are non-increasing (the LCM is concave).
    """

    x_knots: NDArray[np.float64]
    y_knots: NDArray[np.float64]
    slopes: NDArray[np.float64]


def grenander_estimator(
    sorted_pvalues: NDArray[np.float64],
    m_total: int,
) -> GrenanderResult:
    """Compute the Grenander estimator: least concave majorant of the ECDF.

    Parameters
    ----------
    sorted_pvalues : NDArray[np.float64]
        P-values **sorted in ascending order** within a single stratum.
    m_total : int
        Total number of hypotheses in this stratum (may be larger than
        ``len(sorted_pvalues)`` when not all p-values are observed).

    Returns
    -------
    GrenanderResult
        Knot coordinates and slopes of the piecewise-linear LCM.

    Notes
    -----
    Equivalent to ``fdrtool::gcmlcm(x, y, type="lcm")`` in R.

    The ECDF is computed at unique p-value positions, padded with (0, 0)
    and (1, 1) boundary points if needed. The LCM is then computed via
    an upper convex hull algorithm on these points.
    """
    unique_pvalues, counts = np.unique(sorted_pvalues, return_counts=True)
    ecdf_values = np.cumsum(counts) / m_total

    # Pad with (0, 0) at the left if needed
    if unique_pvalues[0] > 0:
        unique_pvalues = np.concatenate(([0.0], unique_pvalues))
        ecdf_values = np.concatenate(([0.0], ecdf_values))

    # Pad with (1, 1) at the right if needed
    if unique_pvalues[-1] < 1.0:
        unique_pvalues = np.concatenate((unique_pvalues, [1.0]))
        ecdf_values = np.concatenate((ecdf_values, [1.0]))

    # Compute least concave majorant via upper convex hull
    # We walk left to right, maintaining a stack of points such that
    # consecutive slopes are non-increasing.
    hull_x = [unique_pvalues[0]]
    hull_y = [ecdf_values[0]]

    for i in range(1, len(unique_pvalues)):
        # Add new point and remove any that violate concavity
        while len(hull_x) >= 2:
            # Check if the last segment's slope >= new segment's slope
            dx1 = hull_x[-1] - hull_x[-2]
            dy1 = hull_y[-1] - hull_y[-2]
            dx2 = unique_pvalues[i] - hull_x[-1]
            dy2 = ecdf_values[i] - hull_y[-1]
            # Cross product: if dy1*dx2 <= dy2*dx1, last point is below the
            # line from [-2] to [i], so remove it (slope is increasing)
            if dy1 * dx2 <= dy2 * dx1:
                hull_x.pop()
                hull_y.pop()
            else:
                break
        hull_x.append(unique_pvalues[i])
        hull_y.append(ecdf_values[i])

    hull_x_arr = np.array(hull_x)
    hull_y_arr = np.array(hull_y)

    # Compute slopes between consecutive knots
    slopes = np.diff(hull_y_arr) / np.diff(hull_x_arr)

    # Return knots without the last point (matching R's gcmlcm convention:
    # x_knots and y_knots are left endpoints; slopes[i] is the slope from
    # knot i to knot i+1)
    return GrenanderResult(
        x_knots=hull_x_arr[:-1],
        y_knots=hull_y_arr[:-1],
        slopes=slopes,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_utils.py::TestGrenanderEstimator -v`
Expected: 5 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/utils.py tests/test_utils.py
uv run ruff check src/pyihw/utils.py tests/test_utils.py
git add src/pyihw/utils.py tests/test_utils.py
git commit -m "Add Grenander estimator (LCM of ECDF) for weight optimization"
```

---

### Task 5: `utils.py` — Weighted Storey pi0 estimator

**Files:**
- Modify: `src/pyihw/utils.py` (add function)
- Modify: `tests/test_utils.py` (add tests)

**Context:** Storey's pi0 estimator, adapted for weighted p-values. Used optionally to adjust weights within each fold. Direct port of R's `weighted_storey_pi0`.

**Step 1: Write the failing tests**

Add to `tests/test_utils.py`:

```python
from pyihw.utils import weighted_storey_pi0


class TestWeightedStoreyPi0:
    def test_all_null(self) -> None:
        """Uniform p-values should give pi0 close to 1."""
        rng = np.random.default_rng(42)
        pvalues = rng.uniform(size=10000)
        weights = np.ones(10000)
        pi0 = weighted_storey_pi0(pvalues, weights)
        assert 0.9 < pi0 < 1.15

    def test_half_signal(self) -> None:
        """With 50% signal, pi0 should be roughly 0.5."""
        rng = np.random.default_rng(42)
        null_p = rng.uniform(size=5000)
        signal_p = rng.beta(0.3, 5, size=5000)
        pvalues = np.concatenate([null_p, signal_p])
        weights = np.ones(10000)
        pi0 = weighted_storey_pi0(pvalues, weights)
        assert 0.3 < pi0 < 0.7

    def test_weighted(self) -> None:
        """Verify the formula: (max(w) + sum(w * (p > tau))) / (m * (1 - tau))."""
        pvalues = np.array([0.1, 0.6, 0.8])
        weights = np.array([2.0, 1.0, 0.5])
        # tau=0.5: w_inf=2.0, sum(w*(p>0.5)) = 1.0 + 0.5 = 1.5
        # pi0 = (2.0 + 1.5) / (3 * 0.5) = 3.5 / 1.5
        expected = 3.5 / 1.5
        result = weighted_storey_pi0(pvalues, weights, tau=0.5, m=3)
        np.testing.assert_allclose(result, expected)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_utils.py::TestWeightedStoreyPi0 -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

Add to `src/pyihw/utils.py`:

```python
def weighted_storey_pi0(
    pvalues: NDArray[np.float64],
    weights: NDArray[np.float64],
    tau: float = 0.5,
    m: int | None = None,
) -> float:
    """Estimate the null proportion using a weighted Storey estimator.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        P-values.
    weights : NDArray[np.float64]
        Per-hypothesis weights.
    tau : float
        Threshold for the estimator. Defaults to 0.5.
    m : int, optional
        Total number of hypotheses. Defaults to ``len(pvalues)``.

    Returns
    -------
    float
        Estimated proportion of null hypotheses.
    """
    if m is None:
        m = len(pvalues)
    w_inf = float(np.max(weights))
    numerator = w_inf + float(np.sum(weights * (pvalues > tau)))
    return numerator / (m * (1.0 - tau))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_utils.py::TestWeightedStoreyPi0 -v`
Expected: 3 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/utils.py tests/test_utils.py
uv run ruff check src/pyihw/utils.py tests/test_utils.py
git add src/pyihw/utils.py tests/test_utils.py
git commit -m "Add weighted Storey pi0 estimator for null proportion estimation"
```

---

### Task 6: `weighting.py` — Threshold-to-weight conversion and penalty functions

**Files:**
- Create: `src/pyihw/weighting.py`
- Test: `tests/test_weighting.py`

**Context:** Before the LP, implement the helper functions: `thresholds_to_weights` converts per-bin thresholds to weights satisfying the weight constraint (mean=1), `total_variation` and `uniform_deviation` compute penalty values for validation.

**Step 1: Write the failing tests**

Create `tests/test_weighting.py`:

```python
from __future__ import annotations

import numpy as np

from pyihw.weighting import thresholds_to_weights, total_variation, uniform_deviation


class TestThresholdsToWeights:
    def test_weight_constraint(self) -> None:
        ts = np.array([0.01, 0.05, 0.02])
        m_groups = np.array([100, 200, 100])
        ws = thresholds_to_weights(ts, m_groups)
        # Weighted mean of weights should be 1
        np.testing.assert_allclose(np.sum(ws * m_groups) / np.sum(m_groups), 1.0)

    def test_all_zero_thresholds(self) -> None:
        ts = np.array([0.0, 0.0, 0.0])
        m_groups = np.array([100, 200, 100])
        ws = thresholds_to_weights(ts, m_groups)
        np.testing.assert_array_equal(ws, [1.0, 1.0, 1.0])

    def test_proportional_to_thresholds(self) -> None:
        ts = np.array([0.02, 0.04])
        m_groups = np.array([100, 100])
        ws = thresholds_to_weights(ts, m_groups)
        # With equal group sizes, w_g = t_g * nbins / sum(t_g)
        expected = ts * 2 / np.sum(ts)
        np.testing.assert_allclose(ws, expected)

    def test_length_mismatch_raises(self) -> None:
        try:
            thresholds_to_weights(np.array([0.1, 0.2]), np.array([100]))
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass


class TestPenalties:
    def test_total_variation_uniform(self) -> None:
        ws = np.array([1.0, 1.0, 1.0])
        assert total_variation(ws) == 0.0

    def test_total_variation_known(self) -> None:
        ws = np.array([0.5, 1.5, 0.5])
        np.testing.assert_allclose(total_variation(ws), 2.0)

    def test_uniform_deviation_uniform(self) -> None:
        ws = np.array([1.0, 1.0, 1.0])
        assert uniform_deviation(ws) == 0.0

    def test_uniform_deviation_known(self) -> None:
        ws = np.array([0.5, 1.5, 1.0])
        np.testing.assert_allclose(uniform_deviation(ws), 1.0)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weighting.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/pyihw/weighting.py`:

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def thresholds_to_weights(
    thresholds: NDArray[np.float64],
    m_groups: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Convert per-bin thresholds to weights satisfying the weight constraint.

    Parameters
    ----------
    thresholds : NDArray[np.float64]
        Rejection threshold for each bin.
    m_groups : NDArray[np.intp]
        Number of hypotheses in each bin.

    Returns
    -------
    NDArray[np.float64]
        Weights ``w_g`` such that the weighted mean equals 1:
        ``sum(w_g * m_g) / sum(m_g) == 1``.

    Notes
    -----
    Formula: ``w_g = t_g * m / sum(m_g * t_g)`` where ``m = sum(m_g)``.
    If all thresholds are zero, returns uniform weights of 1.
    """
    if len(thresholds) != len(m_groups):
        raise ValueError(
            f"Length mismatch: {len(thresholds)} thresholds vs {len(m_groups)} groups"
        )
    if np.all(thresholds == 0.0):
        return np.ones(len(thresholds))
    m = np.sum(m_groups)
    return thresholds * m / np.sum(m_groups * thresholds)


def total_variation(weights: NDArray[np.float64]) -> float:
    """Total variation of a weight vector: ``sum(|w_{g+1} - w_g|)``."""
    return float(np.sum(np.abs(np.diff(weights))))


def uniform_deviation(weights: NDArray[np.float64]) -> float:
    """Uniform deviation of a weight vector: ``sum(|w_g - 1|)``."""
    return float(np.sum(np.abs(weights - 1.0)))
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weighting.py -v`
Expected: 8 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/weighting.py tests/test_weighting.py
uv run ruff check src/pyihw/weighting.py tests/test_weighting.py
git add src/pyihw/weighting.py tests/test_weighting.py
git commit -m "Add thresholds_to_weights, total_variation, uniform_deviation"
```

---

### Task 7: `weighting.py` — `ihw_convex` LP solver

**Files:**
- Modify: `src/pyihw/weighting.py` (add `ihw_convex`)
- Modify: `tests/test_weighting.py` (add tests)

**Context:** This is the core optimization function. Given sorted p-values split by bin, it (1) runs the Grenander estimator per bin, (2) formulates a linear program to maximize rejections subject to the weight constraint and regularization, (3) solves via scipy linprog, (4) converts the solution thresholds to weights. Supports both BH (FDR) and Bonferroni (FWER) constraints, and total variation / uniform deviation penalties.

This is the most complex function. Read the R source in `IHW/R/ihw_convex.R` lines 518-696 carefully for the LP formulation.

**Step 1: Write the failing tests**

Add to `tests/test_weighting.py`:

```python
from pyihw.weighting import ihw_convex


class TestIhwConvex:
    def test_lambda_zero_returns_uniform(self) -> None:
        rng = np.random.default_rng(42)
        split_pvalues = [np.sort(rng.uniform(size=100)) for _ in range(3)]
        m_groups = np.array([100, 100, 100])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=0.0,
            adjustment_type="bh",
        )
        np.testing.assert_array_equal(ws, [1.0, 1.0, 1.0])

    def test_weight_constraint_bh(self) -> None:
        rng = np.random.default_rng(42)
        # Bin 0: mostly signal, Bin 1: mostly null
        signal_p = np.sort(rng.beta(0.3, 5, size=200))
        null_p = np.sort(rng.uniform(size=200))
        split_pvalues = [signal_p, null_p]
        m_groups = np.array([200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bh",
        )
        # Weighted mean should be 1
        np.testing.assert_allclose(np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6)

    def test_weight_constraint_bonferroni(self) -> None:
        rng = np.random.default_rng(42)
        signal_p = np.sort(rng.beta(0.3, 5, size=200))
        null_p = np.sort(rng.uniform(size=200))
        split_pvalues = [signal_p, null_p]
        m_groups = np.array([200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bonferroni",
        )
        np.testing.assert_allclose(np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6)

    def test_signal_bin_gets_higher_weight(self) -> None:
        rng = np.random.default_rng(42)
        signal_p = np.sort(rng.beta(0.3, 5, size=500))
        null_p = np.sort(rng.uniform(size=500))
        split_pvalues = [signal_p, null_p]
        m_groups = np.array([500, 500])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=np.inf,
            adjustment_type="bh",
        )
        # Signal bin should get higher weight
        assert ws[0] > ws[1]

    def test_uniform_deviation_penalty(self) -> None:
        rng = np.random.default_rng(42)
        split_pvalues = [np.sort(rng.beta(0.5, 5, size=200)) for _ in range(3)]
        m_groups = np.array([200, 200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="uniform_deviation",
            lambda_=np.inf,
            adjustment_type="bh",
        )
        np.testing.assert_allclose(np.sum(ws * m_groups) / np.sum(m_groups), 1.0, atol=1e-6)

    def test_weights_are_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        split_pvalues = [np.sort(rng.uniform(size=200)) for _ in range(5)]
        m_groups = np.array([200, 200, 200, 200, 200])
        ws = ihw_convex(
            split_sorted_pvalues=split_pvalues,
            alpha=0.1,
            m_groups=m_groups,
            m_groups_grenander=m_groups,
            penalty="total_variation",
            lambda_=5.0,
            adjustment_type="bh",
        )
        assert np.all(ws >= -1e-10)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_weighting.py::TestIhwConvex -v`
Expected: FAIL — `ImportError`

**Step 3: Write the implementation**

Add to `src/pyihw/weighting.py`:

```python
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from pyihw.utils import grenander_estimator


def ihw_convex(
    split_sorted_pvalues: list[NDArray[np.float64]],
    alpha: float,
    m_groups: NDArray[np.intp],
    m_groups_grenander: NDArray[np.intp],
    penalty: str,
    lambda_: float,
    adjustment_type: str,
) -> NDArray[np.float64]:
    """Solve the IHW convex optimization for per-bin weights.

    Parameters
    ----------
    split_sorted_pvalues : list of NDArray[np.float64]
        Sorted p-values, one array per bin.
    alpha : float
        Significance level.
    m_groups : NDArray[np.intp]
        Hypotheses per bin (used for weight budget).
    m_groups_grenander : NDArray[np.intp]
        Hypotheses per bin (used for Grenander estimator).
    penalty : str
        ``"total_variation"`` or ``"uniform_deviation"``.
    lambda\\_ : float
        Regularization parameter. 0 gives uniform weights, ``np.inf``
        means no regularization.
    adjustment_type : str
        ``"bh"`` or ``"bonferroni"``.

    Returns
    -------
    NDArray[np.float64]
        Weight for each bin, satisfying the weight constraint
        (weighted mean = 1).

    Notes
    -----
    Formulates and solves a linear program to maximize the number of
    rejections. The LP variables are ``(y_1..y_G, t_1..t_G, [aux])``
    where ``y_g`` approximates the CDF at the threshold and ``t_g`` is
    the threshold for bin *g*. Constraints enforce the Grenander
    (concave CDF) bounds, the FDR or FWER budget, and the
    regularization penalty.
    """
    nbins = len(split_sorted_pvalues)

    if lambda_ == 0.0:
        return np.ones(nbins)

    # Clip tiny p-values for numerical stability
    split_sorted_pvalues = [
        np.where(p > 1e-20, p, 0.0) for p in split_sorted_pvalues
    ]

    m = int(np.sum(m_groups))

    # Grenander estimator per bin
    grenander_list = [
        grenander_estimator(p, int(mg))
        for p, mg in zip(split_sorted_pvalues, m_groups_grenander)
    ]

    # --- Build LP ---
    # Variables: [y_1..y_G, t_1..t_G]  (2*nbins)
    # Grenander constraints: y_g + slope_k * t_g <= intercept_k
    # i.e.  y_g - slope_k * t_g <= y_knot_k - slope_k * x_knot_k

    nconstraints_per_bin = [len(g.slopes) for g in grenander_list]
    nconstraints = sum(nconstraints_per_bin)

    # Build constraint matrix rows for Grenander
    rows_i = []
    cols_j = []
    vals_v = []
    rhs = []

    row_offset = 0
    for g_idx, gren in enumerate(grenander_list):
        nk = len(gren.slopes)
        row_indices = np.arange(row_offset, row_offset + nk)
        # y_g coefficient = 1
        rows_i.append(row_indices)
        cols_j.append(np.full(nk, g_idx))
        vals_v.append(np.ones(nk))
        # t_g coefficient = -slope_k
        rows_i.append(row_indices)
        cols_j.append(np.full(nk, nbins + g_idx))
        vals_v.append(-gren.slopes)
        # RHS = y_knot - slope * x_knot
        rhs.append(gren.y_knots - gren.slopes * gren.x_knots)
        row_offset += nk

    # Objective: maximize sum(m_g/m * nbins * y_g)
    # linprog minimizes, so negate
    nvars = 2 * nbins
    obj = np.zeros(nvars)
    obj[:nbins] = -m_groups / m * nbins  # negate for minimization

    # Concatenate Grenander constraints
    all_rows_i = np.concatenate(rows_i)
    all_cols_j = np.concatenate(cols_j)
    all_vals_v = np.concatenate(vals_v)
    all_rhs = np.concatenate(rhs)

    # Regularization auxiliary variables
    if np.isfinite(lambda_):
        if penalty == "total_variation":
            # Auxiliary vars f_1..f_{G-1} for |t_{g+1} - t_g|
            n_aux = nbins - 1
            nvars += n_aux
            obj = np.concatenate([obj, np.zeros(n_aux)])

            # Constraints: t_{g+1} - t_g - f_g <= 0  and  -t_{g+1} + t_g - f_g <= 0
            aux_rows_1_i = []
            aux_rows_1_j = []
            aux_rows_1_v = []
            aux_rows_2_i = []
            aux_rows_2_j = []
            aux_rows_2_v = []

            for k in range(nbins - 1):
                r1 = row_offset + k
                # t_{g+1} - t_g - f_g <= 0
                aux_rows_1_i.extend([r1, r1, r1])
                aux_rows_1_j.extend([nbins + k + 1, nbins + k, 2 * nbins + k])
                aux_rows_1_v.extend([1.0, -1.0, -1.0])
                r2 = row_offset + n_aux + k
                # -t_{g+1} + t_g - f_g <= 0
                aux_rows_2_i.extend([r2, r2, r2])
                aux_rows_2_j.extend([nbins + k + 1, nbins + k, 2 * nbins + k])
                aux_rows_2_v.extend([-1.0, 1.0, -1.0])

            all_rows_i = np.concatenate([all_rows_i, np.array(aux_rows_1_i), np.array(aux_rows_2_i)])
            all_cols_j = np.concatenate([all_cols_j, np.array(aux_rows_1_j), np.array(aux_rows_2_j)])
            all_vals_v = np.concatenate([all_vals_v, np.array(aux_rows_1_v), np.array(aux_rows_2_v)])
            all_rhs = np.concatenate([all_rhs, np.zeros(2 * n_aux)])
            row_offset += 2 * n_aux

            # TV penalty constraint: sum(f_g) - lambda * sum(m_g/m * t_g) <= 0
            tv_row_i = []
            tv_row_j = []
            tv_row_v = []
            r = row_offset
            for k in range(n_aux):
                tv_row_i.append(r)
                tv_row_j.append(2 * nbins + k)
                tv_row_v.append(1.0)
            for k in range(nbins):
                tv_row_i.append(r)
                tv_row_j.append(nbins + k)
                tv_row_v.append(-lambda_ * m_groups[k] / m)
            all_rows_i = np.concatenate([all_rows_i, np.array(tv_row_i)])
            all_cols_j = np.concatenate([all_cols_j, np.array(tv_row_j)])
            all_vals_v = np.concatenate([all_vals_v, np.array(tv_row_v)])
            all_rhs = np.concatenate([all_rhs, [0.0]])
            row_offset += 1

        elif penalty == "uniform_deviation":
            # Auxiliary vars f_1..f_G for |m*t_g - sum(m_i*t_i)|
            n_aux = nbins
            nvars += n_aux
            obj = np.concatenate([obj, np.zeros(n_aux)])

            m_groups_f = m_groups.astype(np.float64)
            for k in range(nbins):
                r1 = row_offset + k
                r2 = row_offset + nbins + k

                # m*t_g - sum(m_i*t_i) - f_g <= 0
                # -m*t_g + sum(m_i*t_i) - f_g <= 0
                for g in range(nbins):
                    coeff1 = -m_groups_f[g]
                    coeff2 = m_groups_f[g]
                    if g == k:
                        coeff1 += m
                        coeff2 -= m
                    if coeff1 != 0:
                        all_rows_i = np.append(all_rows_i, r1)
                        all_cols_j = np.append(all_cols_j, nbins + g)
                        all_vals_v = np.append(all_vals_v, coeff1)
                    if coeff2 != 0:
                        all_rows_i = np.append(all_rows_i, r2)
                        all_cols_j = np.append(all_cols_j, nbins + g)
                        all_vals_v = np.append(all_vals_v, coeff2)

                # -f_g coefficient
                all_rows_i = np.append(all_rows_i, r1)
                all_cols_j = np.append(all_cols_j, 2 * nbins + k)
                all_vals_v = np.append(all_vals_v, -1.0)
                all_rows_i = np.append(all_rows_i, r2)
                all_cols_j = np.append(all_cols_j, 2 * nbins + k)
                all_vals_v = np.append(all_vals_v, -1.0)

            all_rhs = np.concatenate([all_rhs, np.zeros(2 * nbins)])
            row_offset += 2 * nbins

            # UD penalty: sum(f_g) - lambda * sum(m_g * t_g) <= 0
            ud_row_i = []
            ud_row_j = []
            ud_row_v = []
            r = row_offset
            for k in range(nbins):
                ud_row_i.append(r)
                ud_row_j.append(2 * nbins + k)
                ud_row_v.append(1.0)
            for k in range(nbins):
                ud_row_i.append(r)
                ud_row_j.append(nbins + k)
                ud_row_v.append(-lambda_ * m_groups_f[k])
            all_rows_i = np.concatenate([all_rows_i, np.array(ud_row_i)])
            all_cols_j = np.concatenate([all_cols_j, np.array(ud_row_j)])
            all_vals_v = np.concatenate([all_vals_v, np.array(ud_row_v)])
            all_rhs = np.concatenate([all_rhs, [0.0]])
            row_offset += 1

    # FDR/FWER constraint
    fdr_row_i = []
    fdr_row_j = []
    fdr_row_v = []
    r = row_offset
    m_groups_f = m_groups.astype(np.float64)

    if adjustment_type == "bh":
        # sum(m_g * t_g) - alpha * sum(m_g * y_g) <= 0
        for k in range(nbins):
            fdr_row_i.extend([r, r])
            fdr_row_j.extend([k, nbins + k])
            fdr_row_v.extend([-alpha * m_groups_f[k], m_groups_f[k]])
        fdr_rhs = 0.0
    elif adjustment_type == "bonferroni":
        # sum(m_g * t_g) <= alpha
        for k in range(nbins):
            fdr_row_i.append(r)
            fdr_row_j.append(nbins + k)
            fdr_row_v.append(m_groups_f[k])
        fdr_rhs = alpha
    else:
        raise ValueError(f"Unknown adjustment_type: {adjustment_type!r}")

    all_rows_i = np.concatenate([all_rows_i, np.array(fdr_row_i)])
    all_cols_j = np.concatenate([all_cols_j, np.array(fdr_row_j)])
    all_vals_v = np.concatenate([all_vals_v, np.array(fdr_row_v)])
    all_rhs = np.concatenate([all_rhs, [fdr_rhs]])
    row_offset += 1

    # Build sparse constraint matrix
    nrows = row_offset
    A_ub = csc_matrix(
        (all_vals_v, (all_rows_i.astype(int), all_cols_j.astype(int))),
        shape=(nrows, nvars),
    )

    # Variable bounds: y_g in [0, 2], t_g in [0, 2], aux in [0, inf)
    bounds = [(0.0, 2.0)] * (2 * nbins) + [(0.0, None)] * (nvars - 2 * nbins)

    result = linprog(
        c=obj,
        A_ub=A_ub,
        b_ub=all_rhs,
        bounds=bounds,
        method="highs",
    )

    # Extract thresholds, guard against tiny negatives from solver
    ts = np.maximum(result.x[nbins : 2 * nbins], 0.0)
    return thresholds_to_weights(ts, m_groups)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_weighting.py::TestIhwConvex -v`
Expected: 6 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/weighting.py tests/test_weighting.py
uv run ruff check src/pyihw/weighting.py tests/test_weighting.py
git add src/pyihw/weighting.py tests/test_weighting.py
git commit -m "Add ihw_convex LP solver for weight optimization"
```

---

### Task 8: `ihw.py` — `_ihw_internal` k-fold loop

**Files:**
- Create: `src/pyihw/ihw.py`
- Modify: `tests/test_ihw.py` (create)

**Context:** The internal function that orchestrates the k-fold cross-validation. For each fold: hold out data, run nested CV to select lambda, solve `ihw_convex`, optionally apply Storey pi0. This is not the public API yet — tested via an internal integration test.

**Step 1: Write the failing tests**

Create `tests/test_ihw.py`:

```python
from __future__ import annotations

import numpy as np

from pyihw.ihw import _ihw_internal
from pyihw.splitting import assign_folds, groups_by_filter


def _make_test_data(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wasserman simulation: 10000 hypotheses, pi0=0.85, covariate in [0, 3]."""
    m = 10000
    covariates = rng.uniform(0, 3, size=m)
    signals = rng.binomial(1, 0.15, size=m)
    z = rng.normal(loc=signals * covariates)
    from scipy.stats import norm

    pvalues = 1 - norm.cdf(z)
    return pvalues, covariates, signals, z


class TestIhwInternal:
    def test_single_fold_returns_weights(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates, _, _ = _make_test_data(rng)
        groups = groups_by_filter(covariates, nbins=10, rng=np.random.default_rng(1))
        order = np.argsort(pvalues)
        sorted_groups = groups[order]
        sorted_pvalues = pvalues[order]
        m_groups = np.bincount(sorted_groups)

        result = _ihw_internal(
            sorted_groups=sorted_groups,
            sorted_pvalues=sorted_pvalues,
            alpha=0.1,
            lambdas=np.array([0.0, 1.0, 5.0, np.inf]),
            m_groups=m_groups,
            penalty="total_variation",
            nfolds=1,
            nfolds_internal=1,
            nsplits_internal=1,
            adjustment_type="bh",
            null_proportion=False,
            null_proportion_level=0.5,
            rng=np.random.default_rng(2),
        )
        # Weight budget
        np.testing.assert_allclose(result["sorted_weights"].sum(), len(pvalues), atol=1.0)

    def test_multi_fold_weight_budget(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates, _, _ = _make_test_data(rng)
        groups = groups_by_filter(covariates, nbins=10, rng=np.random.default_rng(1))
        order = np.argsort(pvalues)
        sorted_groups = groups[order]
        sorted_pvalues = pvalues[order]
        m_groups = np.bincount(sorted_groups)

        result = _ihw_internal(
            sorted_groups=sorted_groups,
            sorted_pvalues=sorted_pvalues,
            alpha=0.1,
            lambdas=np.array([0.0, 5.0, np.inf]),
            m_groups=m_groups,
            penalty="total_variation",
            nfolds=5,
            nfolds_internal=5,
            nsplits_internal=1,
            adjustment_type="bh",
            null_proportion=False,
            null_proportion_level=0.5,
            rng=np.random.default_rng(2),
        )
        assert result["rjs"] > 0
        assert result["weight_matrix"].shape == (10, 5)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ihw.py::TestIhwInternal -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError`

**Step 3: Write the implementation**

Create `src/pyihw/ihw.py`:

```python
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyihw.splitting import assign_folds
from pyihw.utils import bh_adjust, safe_divide, weighted_storey_pi0
from pyihw.weighting import ihw_convex


def _ihw_internal(
    sorted_groups: NDArray[np.intp],
    sorted_pvalues: NDArray[np.float64],
    alpha: float,
    lambdas: NDArray[np.float64],
    m_groups: NDArray[np.intp],
    penalty: str,
    nfolds: int,
    nfolds_internal: int,
    nsplits_internal: int,
    adjustment_type: str,
    null_proportion: bool,
    null_proportion_level: float,
    rng: np.random.Generator,
    sorted_folds: NDArray[np.intp] | None = None,
) -> dict[str, Any]:
    """K-fold cross-validated IHW on pre-sorted data.

    Parameters
    ----------
    sorted_groups : NDArray[np.intp]
        Bin assignments, ordered by ascending p-value.
    sorted_pvalues : NDArray[np.float64]
        P-values sorted in ascending order.
    alpha : float
        Significance level.
    lambdas : NDArray[np.float64]
        Grid of regularization parameters to search over.
    m_groups : NDArray[np.intp]
        Total hypotheses per bin.
    penalty : str
        ``"total_variation"`` or ``"uniform_deviation"``.
    nfolds : int
        Number of outer folds.
    nfolds_internal : int
        Number of inner folds for lambda selection.
    nsplits_internal : int
        Number of repeated inner splits.
    adjustment_type : str
        ``"bh"`` or ``"bonferroni"``.
    null_proportion : bool
        Whether to apply Storey's pi0 estimator.
    null_proportion_level : float
        Threshold tau for Storey's estimator.
    rng : numpy.random.Generator
        Random number generator.
    sorted_folds : NDArray[np.intp], optional
        Pre-specified fold assignments (sorted by p-value order).

    Returns
    -------
    dict
        Keys: ``fold_lambdas``, ``rjs``, ``sorted_pvalues``,
        ``sorted_weighted_pvalues``, ``sorted_adj_p``,
        ``sorted_weights``, ``sorted_groups``, ``sorted_folds``,
        ``weight_matrix``.
    """
    m = len(sorted_pvalues)
    nbins = len(m_groups)

    folds_prespecified = sorted_folds is not None
    if sorted_folds is None:
        sorted_folds = assign_folds(m, nfolds, rng)

    sorted_weights = np.full(m, np.nan)
    fold_lambdas = np.full(nfolds, np.nan)
    weight_matrix = np.full((nbins, nfolds), np.nan)

    for i in range(nfolds):
        if nfolds == 1:
            mask_train = np.ones(m, dtype=bool)
            m_groups_holdout = m_groups.copy()
            m_groups_train = m_groups.copy()
        else:
            mask_train = sorted_folds != i
            train_groups = sorted_groups[mask_train]
            train_pvalues = sorted_pvalues[mask_train]

            # Count available p-values per group in training set
            train_group_counts = np.bincount(train_groups, minlength=nbins)

            if not folds_prespecified:
                available_per_group = np.bincount(sorted_groups, minlength=nbins)
                m_groups_holdout = (
                    (m_groups - available_per_group) / nfolds
                    + available_per_group
                    - train_group_counts
                )
                m_groups_train = m_groups - m_groups_holdout
            else:
                # For prespecified folds, we'd need m_groups as a matrix
                # For simplicity, estimate from data
                holdout_group_counts = np.bincount(
                    sorted_groups[sorted_folds == i], minlength=nbins
                )
                m_groups_holdout = holdout_group_counts.astype(np.float64)
                m_groups_train = train_group_counts.astype(np.float64)

        if nfolds == 1:
            filtered_sorted_groups = sorted_groups
            filtered_sorted_pvalues = sorted_pvalues
        else:
            filtered_sorted_groups = train_groups
            filtered_sorted_pvalues = train_pvalues

        # Split p-values by group (sorted within each group)
        filtered_split = [
            np.sort(filtered_sorted_pvalues[filtered_sorted_groups == g])
            for g in range(nbins)
        ]

        # Lambda selection via nested CV
        if len(lambdas) > 1:
            rjs_matrix = np.zeros((len(lambdas), nsplits_internal))
            for k, lam in enumerate(lambdas):
                for s in range(nsplits_internal):
                    inner_result = _ihw_internal(
                        sorted_groups=filtered_sorted_groups,
                        sorted_pvalues=filtered_sorted_pvalues,
                        alpha=alpha,
                        lambdas=np.array([lam]),
                        m_groups=m_groups_train.astype(np.intp),
                        penalty=penalty,
                        nfolds=nfolds_internal,
                        nfolds_internal=nfolds_internal,
                        nsplits_internal=1,
                        adjustment_type=adjustment_type,
                        null_proportion=False,
                        null_proportion_level=null_proportion_level,
                        rng=rng,
                    )
                    rjs_matrix[k, s] = inner_result["rjs"]
            best_lambda = lambdas[np.argmax(rjs_matrix.mean(axis=1))]
        else:
            best_lambda = lambdas[0]

        fold_lambdas[i] = best_lambda

        # Solve for weights with chosen lambda
        ws = ihw_convex(
            split_sorted_pvalues=filtered_split,
            alpha=alpha,
            m_groups=m_groups_holdout.astype(np.intp),
            m_groups_grenander=m_groups_train.astype(np.intp),
            penalty=penalty,
            lambda_=float(best_lambda),
            adjustment_type=adjustment_type,
        )

        # Assign weights to hypotheses in this fold
        fold_mask = sorted_folds == i
        sorted_weights[fold_mask] = ws[sorted_groups[fold_mask]]
        weight_matrix[:, i] = ws

        # Optional null proportion adjustment
        if null_proportion:
            pi0_est = weighted_storey_pi0(
                sorted_pvalues[fold_mask],
                sorted_weights[fold_mask],
                tau=null_proportion_level,
                m=int(np.sum(m_groups_holdout)),
            )
            sorted_weights[fold_mask] /= pi0_est
            weight_matrix[:, i] /= pi0_est

    sorted_weighted_pvalues = safe_divide(sorted_pvalues, sorted_weights)

    m_total = int(np.sum(m_groups))
    if adjustment_type == "bh":
        sorted_adj_p = bh_adjust(sorted_weighted_pvalues, m_total=m_total)
    elif adjustment_type == "bonferroni":
        sorted_adj_p = np.minimum(sorted_weighted_pvalues * m_total, 1.0)
    else:
        raise ValueError(f"Unknown adjustment_type: {adjustment_type!r}")

    rjs = int(np.sum(sorted_adj_p <= alpha))

    return {
        "fold_lambdas": fold_lambdas,
        "rjs": rjs,
        "sorted_pvalues": sorted_pvalues,
        "sorted_weighted_pvalues": sorted_weighted_pvalues,
        "sorted_adj_p": sorted_adj_p,
        "sorted_weights": sorted_weights,
        "sorted_groups": sorted_groups,
        "sorted_folds": sorted_folds,
        "weight_matrix": weight_matrix,
    }
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ihw.py::TestIhwInternal -v`
Expected: 2 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/ihw.py tests/test_ihw.py
uv run ruff check src/pyihw/ihw.py tests/test_ihw.py
git add src/pyihw/ihw.py tests/test_ihw.py
git commit -m "Add _ihw_internal k-fold cross-validation loop"
```

---

### Task 9: `ihw.py` — Public `ihw()` entry point

**Files:**
- Modify: `src/pyihw/ihw.py` (add `ihw` function)
- Modify: `tests/test_ihw.py` (add end-to-end tests)

**Context:** The public API function. Validates inputs, handles binning, sorts data, calls `_ihw_internal`, reorders results, and returns an `IHWResult`.

**Step 1: Write the failing tests**

Add to `tests/test_ihw.py`:

```python
from pyihw.ihw import ihw
from pyihw._types import IHWResult
from pyihw.utils import bh_threshold


def _wasserman_sim(
    rng: np.random.Generator, m: int = 10000, pi0: float = 0.85
) -> tuple[np.ndarray, np.ndarray]:
    covariates = rng.uniform(0, 3, size=m)
    signals = rng.binomial(1, 1 - pi0, size=m)
    z = rng.normal(loc=signals * covariates)
    from scipy.stats import norm

    pvalues = 1 - norm.cdf(z)
    return pvalues, covariates


class TestIhwPublic:
    def test_returns_ihw_result(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        assert isinstance(result, IHWResult)

    def test_more_rejections_than_bh(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        t_bh = bh_threshold(pvalues, alpha=0.1)
        bh_rejections = int(np.sum(pvalues <= t_bh))
        assert result.n_rejections >= bh_rejections

    def test_weight_budget(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        np.testing.assert_allclose(result.weights.sum(), len(pvalues), atol=5.0)

    def test_lower_alpha_fewer_rejections(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        r1 = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        r2 = ihw(pvalues, covariates, alpha=0.01, rng=np.random.default_rng(1))
        assert r2.n_rejections < r1.n_rejections

    def test_single_bin_reduces_to_bh(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng, m=500)
        result = ihw(pvalues, covariates, alpha=0.1, nbins=1, rng=np.random.default_rng(1))
        assert result.nbins == 1
        assert result.nfolds == 1
        # Weights should all be 1
        np.testing.assert_allclose(result.weights, 1.0)

    def test_bonferroni_mode(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng)
        result = ihw(
            pvalues, covariates, alpha=0.1,
            adjustment_type="bonferroni",
            rng=np.random.default_rng(1),
        )
        assert result.adjustment_type == "bonferroni"
        # Bonferroni should have fewer rejections than BH
        result_bh = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        assert result.n_rejections <= result_bh.n_rejections

    def test_output_lengths_match_input(self) -> None:
        rng = np.random.default_rng(42)
        pvalues, covariates = _wasserman_sim(rng, m=5000)
        result = ihw(pvalues, covariates, alpha=0.1, rng=np.random.default_rng(1))
        assert len(result.pvalues) == 5000
        assert len(result.adj_pvalues) == 5000
        assert len(result.weights) == 5000
        assert len(result.weighted_pvalues) == 5000
        assert len(result.groups) == 5000
        assert len(result.folds) == 5000
        assert len(result.covariates) == 5000

    def test_invalid_pvalues_raises(self) -> None:
        try:
            ihw(np.array([-0.1, 0.5]), np.array([1.0, 2.0]), alpha=0.1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_nan_pvalues_raises(self) -> None:
        try:
            ihw(np.array([np.nan, 0.5]), np.array([1.0, 2.0]), alpha=0.1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_mismatched_lengths_raises(self) -> None:
        try:
            ihw(np.array([0.1, 0.5]), np.array([1.0]), alpha=0.1)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_invalid_alpha_raises(self) -> None:
        try:
            ihw(np.array([0.1, 0.5]), np.array([1.0, 2.0]), alpha=1.5)
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ihw.py::TestIhwPublic -v`
Expected: FAIL — `ImportError: cannot import name 'ihw'`

**Step 3: Write the implementation**

Add to `src/pyihw/ihw.py` (above `_ihw_internal`):

```python
from pyihw._types import IHWResult
from pyihw.splitting import groups_by_filter


def ihw(
    pvalues: NDArray[np.float64],
    covariates: NDArray[np.float64],
    alpha: float,
    *,
    covariate_type: str = "ordinal",
    nbins: int | str = "auto",
    nfolds: int = 5,
    nfolds_internal: int = 5,
    nsplits_internal: int = 1,
    lambdas: NDArray[np.float64] | str = "auto",
    adjustment_type: str = "bh",
    null_proportion: bool = False,
    null_proportion_level: float = 0.5,
    folds: NDArray[np.intp] | None = None,
    rng: np.random.Generator | None = None,
) -> IHWResult:
    """Independent Hypothesis Weighting.

    Given p-values and an independent covariate, learn data-driven weights
    to improve power in multiple testing while controlling FDR or FWER.

    Parameters
    ----------
    pvalues : NDArray[np.float64]
        Unadjusted p-values, one per hypothesis.
    covariates : NDArray[np.float64]
        Independent covariate for each hypothesis.
    alpha : float
        Nominal significance level, must be in (0, 1).
    covariate_type : str
        ``"ordinal"`` or ``"nominal"``.
    nbins : int or "auto"
        Number of bins. ``"auto"`` uses ``max(1, min(40, n // 1500))``.
    nfolds : int
        Number of cross-validation folds.
    nfolds_internal : int
        Number of inner folds for regularization parameter selection.
    nsplits_internal : int
        Number of repeated inner splits.
    lambdas : NDArray or "auto"
        Grid of regularization parameters. ``"auto"`` generates a
        default grid.
    adjustment_type : str
        ``"bh"`` for FDR control or ``"bonferroni"`` for FWER control.
    null_proportion : bool
        Whether to estimate and adjust for the null proportion.
    null_proportion_level : float
        Threshold for the Storey pi0 estimator.
    folds : NDArray[np.intp], optional
        Pre-specified fold assignments.
    rng : numpy.random.Generator, optional
        Random number generator. Defaults to ``default_rng(1)``.

    Returns
    -------
    IHWResult
        Result object with adjusted p-values, weights, and metadata.

    Raises
    ------
    ValueError
        If inputs are invalid (NaN p-values, out-of-range alpha, etc.).
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)

    # --- Validation ---
    if np.any(np.isnan(pvalues)):
        raise ValueError("p-values must not contain NaN")
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must be in [0, 1]")
    if len(pvalues) != len(covariates):
        raise ValueError(
            f"Length mismatch: {len(pvalues)} p-values vs {len(covariates)} covariates"
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if np.any(np.isnan(covariates)):
        raise ValueError("Covariates must not contain NaN")
    if adjustment_type not in ("bh", "bonferroni"):
        raise ValueError(f"adjustment_type must be 'bh' or 'bonferroni', got {adjustment_type!r}")

    if rng is None:
        rng = np.random.default_rng(1)

    n = len(pvalues)

    if folds is not None:
        nfolds = len(np.unique(folds))

    # --- Binning ---
    if nbins == "auto":
        nbins = max(1, min(40, n // 1500))

    nbins = int(nbins)

    if covariate_type == "ordinal":
        groups = groups_by_filter(covariates, nbins, rng=rng)
        penalty = "total_variation"
    elif covariate_type == "nominal":
        groups = groups_by_filter(covariates, nbins, rng=rng)
        penalty = "uniform_deviation"
    else:
        raise ValueError(f"covariate_type must be 'ordinal' or 'nominal', got {covariate_type!r}")

    # --- Lambda grid ---
    if isinstance(lambdas, str) and lambdas == "auto":
        lambdas = np.array([0.0, 1.0, nbins / 8, nbins / 4, nbins / 2, nbins, np.inf])

    lambdas = np.asarray(lambdas, dtype=np.float64)

    if nbins < 1:
        raise ValueError("nbins must be >= 1")

    # --- Single bin shortcut ---
    if nbins == 1:
        nfolds = 1
        m_total = n
        if adjustment_type == "bh":
            adj_p = bh_adjust(pvalues, m_total=m_total)
        else:
            adj_p = np.minimum(pvalues * m_total, 1.0)

        return IHWResult(
            pvalues=pvalues,
            adj_pvalues=adj_p,
            weights=np.ones(n),
            weighted_pvalues=pvalues.copy(),
            covariates=covariates,
            groups=np.zeros(n, dtype=np.intp),
            folds=np.zeros(n, dtype=np.intp),
            weight_matrix=np.ones((1, 1)),
            alpha=alpha,
            nbins=1,
            nfolds=1,
            regularization_terms=np.array([0.0]),
            m_groups=np.array([n], dtype=np.intp),
            penalty=penalty,
            covariate_type=covariate_type,
            adjustment_type=adjustment_type,
        )

    # --- Sort by p-value ---
    order = np.argsort(pvalues)
    reorder = np.argsort(order)

    sorted_groups = groups[order]
    sorted_pvalues = pvalues[order]
    sorted_folds = folds[order] if folds is not None else None

    m_groups = np.bincount(sorted_groups, minlength=nbins).astype(np.intp)

    # --- Run k-fold IHW ---
    result = _ihw_internal(
        sorted_groups=sorted_groups,
        sorted_pvalues=sorted_pvalues,
        alpha=alpha,
        lambdas=lambdas,
        m_groups=m_groups,
        penalty=penalty,
        nfolds=nfolds,
        nfolds_internal=nfolds_internal,
        nsplits_internal=nsplits_internal,
        adjustment_type=adjustment_type,
        null_proportion=null_proportion,
        null_proportion_level=null_proportion_level,
        rng=rng,
        sorted_folds=sorted_folds,
    )

    # --- Reorder back to original input order ---
    return IHWResult(
        pvalues=pvalues,
        adj_pvalues=result["sorted_adj_p"][reorder],
        weights=result["sorted_weights"][reorder],
        weighted_pvalues=result["sorted_weighted_pvalues"][reorder],
        covariates=covariates,
        groups=groups,
        folds=result["sorted_folds"][reorder],
        weight_matrix=result["weight_matrix"],
        alpha=alpha,
        nbins=nbins,
        nfolds=nfolds,
        regularization_terms=result["fold_lambdas"],
        m_groups=m_groups,
        penalty=penalty,
        covariate_type=covariate_type,
        adjustment_type=adjustment_type,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ihw.py::TestIhwPublic -v`
Expected: 11 passed

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/ihw.py tests/test_ihw.py
uv run ruff check src/pyihw/ihw.py tests/test_ihw.py
git add src/pyihw/ihw.py tests/test_ihw.py
git commit -m "Add public ihw() entry point with input validation"
```

---

### Task 10: `__init__.py` — Public API exports

**Files:**
- Modify: `src/pyihw/__init__.py`
- Test: Quick smoke test

**Step 1: Write the failing test**

Add to `tests/test_ihw.py`:

```python
def test_public_imports() -> None:
    from pyihw import IHWResult, bh_threshold, ihw

    assert callable(ihw)
    assert callable(bh_threshold)
    assert IHWResult is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ihw.py::test_public_imports -v`
Expected: FAIL — `ImportError`

**Step 3: Update `__init__.py`**

Replace `src/pyihw/__init__.py` contents with:

```python
from __future__ import annotations

from pyihw._types import IHWResult
from pyihw.ihw import ihw
from pyihw.utils import bh_threshold

__all__ = ["ihw", "IHWResult", "bh_threshold"]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ihw.py::test_public_imports -v`
Expected: PASS

**Step 5: Format, lint, commit**

```bash
uv run ruff format src/pyihw/__init__.py
uv run ruff check src/pyihw/__init__.py
git add src/pyihw/__init__.py tests/test_ihw.py
git commit -m "Set up public API exports: ihw, IHWResult, bh_threshold"
```

---

### Task 11: Full test suite pass and cleanup

**Files:**
- All source and test files

**Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests pass.

**Step 2: Format and lint everything**

```bash
uv run ruff format .
uv run ruff check .
```

Fix any issues.

**Step 3: Type check**

```bash
uv run ty check
```

Fix any type errors.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "Fix lint and type errors across all modules"
```

---

### Task 12: Run end-to-end verification

**Step 1: Run full suite with verbose output**

```bash
uv run pytest -v --tb=long
```

**Step 2: Verify no regressions**

All tests should pass. Check that:
- Weight constraint holds in all tests
- IHW produces more rejections than BH on the simulation
- Edge cases (single bin, validation errors) work correctly

**Step 3: Final commit if needed**

```bash
git status
# If any changes: commit them
```
