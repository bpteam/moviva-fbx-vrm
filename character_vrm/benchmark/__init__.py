"""Ground-truth benchmark for the anatomy analyzer and Auto-Author, built from authored VRM files (no Blender)."""

from .runner import obfuscate, run_model, run_pass
from .truth import Truth, read_truth

__all__ = ["Truth", "obfuscate", "read_truth", "run_model", "run_pass"]
