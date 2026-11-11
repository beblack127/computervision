import os
import sys

import cv2
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def write_synthetic_video(path, n_frames=90, h=240, w=320, fps=30, seed=0):
    """Write a small mp4 whose frames differ (moving square + per-frame noise)."""
    rng = np.random.default_rng(seed)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for t in range(n_frames):
        fr = (rng.integers(0, 40, size=(h, w, 3))).astype(np.uint8)
        x = int((w - 40) * t / max(n_frames - 1, 1))
        fr[100:140, x:x + 40] = (255, 128 + (t % 100), 30)
        vw.write(fr)
    vw.release()


@pytest.fixture(scope="session")
def synth_videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("videos")
    specs = {"video0": 90, "video1": 47, "video2": 300, "video3": 9, "video4": 61, "video5": 13}
    for i, (vid, n) in enumerate(specs.items()):
        write_synthetic_video(str(d / f"{vid}.mp4"), n_frames=n, seed=i)
    return str(d), specs
