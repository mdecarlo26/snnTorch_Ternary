"""
Author: Marc DeCarlo, Suman Kumar
Email: marcadecarlo@gmail.com
Date: November 25, 2025
"""
from .neurons import TernaryLeaky
from .spikegen import rate_ternary
from .surrogate import atan_ternary

__all__ = ["TernaryLeaky", "ternary_rate", "atan_ternary"]


def _auto_patch_snntorch():
    """Try to attach ternary extensions into snntorch if it’s installed."""
    try:
        import snntorch as snn
        from snntorch import spikegen as _spikegen
        from snntorch import surrogate as _surrogate
    except ImportError:
        # snntorch not installed or import order weird; just skip
        return

    if not hasattr(snn, "TernaryLeaky"):
        snn.TernaryLeaky = TernaryLeaky

    if not hasattr(_spikegen, "ternary_rate"):
        _spikegen.ternary_rate = rate_ternary

    if not hasattr(_surrogate, "atan_ternary"):
        _surrogate.atan_ternary = atan_ternary


def register_into_snntorch():
    """Explicit API for explicit control"""
    _auto_patch_snntorch()

# run on import to link into snntorch if possible
_auto_patch_snntorch()
