from __future__ import annotations

import io
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray


def load_airway() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load the bundled airway DESeq2 results for IHW demonstration.

    Returns p-values and baseMean values from a differential expression
    analysis of the airway RNA-seq dataset (Himes et al. 2014) using
    DESeq2 with design ``~ cell + dex``. Genes with NA p-values are
    excluded, leaving 33,469 hypotheses.

    Parameters
    ----------
    None

    Returns
    -------
    pvalues : NDArray[np.float64]
        Array of 33,469 p-values from DESeq2.
    basemean : NDArray[np.float64]
        Array of 33,469 baseMean values (mean of normalized counts).

    Notes
    -----
    The airway dataset contains RNA-seq read counts from four human
    airway smooth muscle cell lines, each treated with dexamethasone
    or left untreated. The data is from Himes et al. (2014),
    "RNA-Seq Transcriptome Profiling Identifies CRISPLD2 as a
    Glucocorticoid Responsive Gene", *PLoS ONE* 9(6): e99625.
    GEO accession: GSE52778.

    The baseMean covariate is a strong candidate for IHW because
    genes with higher expression have more statistical power to
    detect differential expression, creating a covariate-power
    relationship that IHW can exploit.
    """
    data_file = files(__package__).joinpath("airway.csv")
    data = np.loadtxt(
        io.StringIO(data_file.read_text(encoding="utf-8")),
        delimiter=",",
        skiprows=1,
    )
    return data[:, 0].copy(), data[:, 1].copy()
