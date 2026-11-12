"""Make the pinned official X-Pool code importable.

Usage:
    from tvr.xpool_bridge import XPOOL_ROOT, ensure_xpool_on_path
    ensure_xpool_on_path()
    from model.clip_transformer import CLIPTransformer   # official module
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XPOOL_ROOT = os.environ.get("XPOOL_ROOT", os.path.join(REPO_ROOT, "third_party", "xpool"))
XPOOL_DATA = os.path.join(XPOOL_ROOT, "data", "MSRVTT")
PINNED_COMMIT = "bc88718199007d395136adc99a3e69adbee874a0"


def ensure_xpool_on_path():
    if not os.path.isdir(os.path.join(XPOOL_ROOT, "model")):
        raise FileNotFoundError(
            f"Official X-Pool code not found at {XPOOL_ROOT}. Run: bash scripts/setup_xpool.sh"
        )
    if XPOOL_ROOT not in sys.path:
        sys.path.insert(0, XPOOL_ROOT)
    return XPOOL_ROOT


def xpool_commit():
    head = os.path.join(XPOOL_ROOT, ".git", "HEAD")
    try:
        with open(head) as f:
            ref = f.read().strip()
        if ref.startswith("ref:"):
            ref_path = os.path.join(XPOOL_ROOT, ".git", ref.split(" ", 1)[1])
            with open(ref_path) as f:
                return f.read().strip()
        return ref
    except OSError:
        return "unknown"
