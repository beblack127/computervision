"""Make the pinned official X-Pool code importable.

Usage:
    from tvr.xpool_bridge import XPOOL_ROOT, ensure_xpool_on_path
    ensure_xpool_on_path()
    from model.clip_transformer import CLIPTransformer   # official module
"""
import importlib.machinery
import os
import sys
import types

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XPOOL_ROOT = os.environ.get("XPOOL_ROOT", os.path.join(REPO_ROOT, "third_party", "xpool"))
XPOOL_DATA = os.path.join(XPOOL_ROOT, "data", "MSRVTT")
PINNED_COMMIT = "bc88718199007d395136adc99a3e69adbee874a0"

# Top-level folders of the official code. None of them has an __init__.py, so Python treats them
# as *namespace* packages, and a regular package with the same name anywhere on sys.path wins
# regardless of path order. On Colab, HuggingFace `datasets` is pre-installed and shadowed
# `datasets.msrvtt_dataset` -> ModuleNotFoundError. We therefore bind these names explicitly.
XPOOL_PACKAGES = ("config", "datasets", "model", "modules", "trainer")


def _bind_package(name, path):
    mod = sys.modules.get(name)
    if mod is not None and list(getattr(mod, "__path__", [])) == [path]:
        return
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]  # drop a shadowing package (e.g. HF `datasets`) and its submodules
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [path]
    mod = types.ModuleType(name)
    mod.__spec__, mod.__path__, mod.__package__ = spec, [path], name
    sys.modules[name] = mod


def ensure_xpool_on_path():
    if not os.path.isdir(os.path.join(XPOOL_ROOT, "model")):
        raise FileNotFoundError(
            f"Official X-Pool code not found at {XPOOL_ROOT}. Run: bash scripts/setup_xpool.sh"
        )
    if XPOOL_ROOT not in sys.path:
        sys.path.insert(0, XPOOL_ROOT)
    for name in XPOOL_PACKAGES:
        path = os.path.join(XPOOL_ROOT, name)
        if os.path.isdir(path):
            _bind_package(name, path)
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
