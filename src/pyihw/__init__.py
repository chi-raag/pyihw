"""pyIHW: Python implementation of Independent Hypothesis Weighting."""

from __future__ import annotations

from pyihw._types import IHWResult
from pyihw.data import load_airway
from pyihw.ihw import ihw
from pyihw.utils import bh_threshold

__all__ = ["IHWResult", "bh_threshold", "ihw", "load_airway"]
