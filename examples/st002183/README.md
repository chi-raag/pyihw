# ST002183 Example

This example uses the public processed LC-MS feature tables from
Metabolomics Workbench study
[ST002183](https://metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID=ST002183)
to exercise `pyihw` on a realistic metabolomics design with biological
samples, pooled QCs, and extraction blanks.

The script downloads the two public analysis tables on first run and
caches them under `examples/st002183/.cache`:

- `AN003575`: reversed-phase, positive mode
- `AN003576`: HILIC, negative mode

## What It Does

`run_analysis.py`:

- pulls the processed `untarg_data` table and compact analysis metadata
- builds one of three contrasts:
  - `between-groups`: `exercise` vs `waitlist` at one time point
  - `paired-within-group`: matched within-subject baseline vs follow-up
  - `change-score`: matched within-subject change, then `exercise` vs `waitlist`
- computes raw p-values from simple t-tests on `log1p(max(intensity, 0))`
- derives IHW covariates from the public matrix:
  - `mean_abundance`
  - `bio_detection_rate`
  - `qc_detection_rate`
  - `qc_rsd_score`
  - `blank_exclusion_score`
  - `combined_quality`
- writes one summary CSV and one per-feature CSV for each analysis/contrast

## Quick Start

Run the default cross-sectional comparison on the HILIC negative table:

```bash
uv run python examples/st002183/run_analysis.py
```

Run both analysis modes for the same contrast:

```bash
uv run python examples/st002183/run_analysis.py --analysis-id all
```

Run the more design-aware change-score comparison from baseline to time
point `1`:

```bash
uv run python examples/st002183/run_analysis.py \
  --contrast change-score \
  --baseline - \
  --followup 1
```

Run a within-group paired comparison:

```bash
uv run python examples/st002183/run_analysis.py \
  --contrast paired-within-group \
  --group exercise \
  --baseline - \
  --followup 1
```

## Notes

- These matrices are small by IHW standards, so `nbins="auto"` would
  collapse to one bin and effectively reduce IHW to BH. The example
  therefore uses `--nbins 8` by default.
- `AN003576` contains a handful of negative intensities in the public
  processed table. The script clips those to zero before `log1p`.
- The default statistics are deliberately simple. This example is meant
  to expose realistic metabolomics covariates and public study plumbing,
  not to claim an optimal DA model for the trial.
