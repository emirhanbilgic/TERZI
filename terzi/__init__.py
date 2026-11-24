"""TERZI: Three-dimensional Garment Generation and Retargeting via Optimization-free Inference."""
import os as _os
import sys as _sys

_T2 = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "TRELLIS.2")
if _os.path.isdir(_T2) and _T2 not in _sys.path:       # the TRELLIS.2 submodule (git submodule update --init)
    _sys.path.append(_T2)

from .body import Body, body_from_smplx_params, load_body          # noqa: E402
from .pipeline import MODELS, TerziOutput, TerziPipeline              # noqa: E402
from .postprocess import fill_holes, patch_holes                     # noqa: E402

__all__ = ["Body", "MODELS", "TerziOutput", "TerziPipeline", "body_from_smplx_params", "fill_holes", "load_body", "patch_holes"]
