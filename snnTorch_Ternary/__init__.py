from .neurons import TernaryLeaky
from .spikegen import ternary_rate

__all__ = ["TernaryLeaky", "ternary_rate"]


def _auto_patch_snntorch():
    """Try to attach ternary extensions into snntorch if it’s installed."""
    try:
        import snntorch as snn
        from snntorch import spikegen as _spikegen
    except ImportError:
        # snntorch not installed or import order weird; just skip
        return

    if not hasattr(snn, "TernaryLeaky"):
        snn.TernaryLeaky = TernaryLeaky

    if not hasattr(_spikegen, "ternary_rate"):
        _spikegen.ternary_rate = ternary_rate


def register_into_snntorch():
    """Explicit API for explicit control"""
    _auto_patch_snntorch()

# run on import to link into snntorch if possible
_auto_patch_snntorch()
