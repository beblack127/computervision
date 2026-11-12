"""Documents (and guards) the torchvision antialias behaviour change."""
import torch

from tvr.compat import init_transform_dict
from tvr.xpool_bridge import ensure_xpool_on_path

ensure_xpool_on_path()
from datasets.model_transforms import init_transform_dict as official_tfms  # noqa: E402


def test_test_transform_has_no_antialias():
    tf = init_transform_dict(224)["clip_test"]
    assert tf.transforms[0].antialias is False


def test_official_transform_differs_on_modern_torchvision():
    x = torch.rand(12, 3, 240, 320, generator=torch.Generator().manual_seed(0))
    ours = init_transform_dict(224)["clip_test"](x)
    official_now = official_tfms(224)["clip_test"](x)
    assert ours.shape == official_now.shape == (12, 3, 224, 224)
    # On torchvision>=0.17 the official code would silently antialias -> not the 2022 behaviour.
    if official_now.__class__ is torch.Tensor and official_tfms(224)["clip_test"].transforms[0].antialias:
        assert (ours - official_now).abs().max() > 1e-3
