"""Official X-Pool folders must win over same-named installed packages (Colab ships HF `datasets`)."""
import importlib
import os
import sys

from tvr.xpool_bridge import XPOOL_ROOT, ensure_xpool_on_path


def test_official_datasets_wins_over_shadowing_package(tmp_path, monkeypatch):
    fake = tmp_path / "datasets"
    fake.mkdir()
    (fake / "__init__.py").write_text("FAKE_HF_DATASETS = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for key in [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]:
        monkeypatch.delitem(sys.modules, key)
    importlib.invalidate_caches()
    import datasets  # the shadowing regular package is imported first, as on Colab
    assert getattr(datasets, "FAKE_HF_DATASETS", False)

    ensure_xpool_on_path()
    from datasets.video_capture import VideoCapture  # official module
    from datasets.msrvtt_dataset import MSRVTTDataset  # noqa: F401

    assert VideoCapture.__module__ == "datasets.video_capture"
    assert sys.modules["datasets"].__path__ == [os.path.join(XPOOL_ROOT, "datasets")]
    assert sys.modules["datasets.video_capture"].__file__.startswith(XPOOL_ROOT)
