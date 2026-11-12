"""The frame cache must be bit-identical to the official on-the-fly decoding."""
import os

import torch

from tvr import frame_cache
from tvr.xpool_bridge import ensure_xpool_on_path

ensure_xpool_on_path()
from datasets.video_capture import VideoCapture  # noqa: E402  (official)


def test_indices_match_official_logic():
    import numpy as np
    for vlen in [1, 5, 11, 12, 13, 47, 300, 901]:
        acc = min(12, vlen)
        intervals = np.linspace(0, vlen, acc + 1).astype(int)
        ranges = [(iv, intervals[i + 1] - 1) for i, iv in enumerate(intervals[:-1])]
        assert frame_cache.uniform_indices(vlen) == [(a + b) // 2 for a, b in ranges]


def test_cache_is_bit_identical(synth_videos, tmp_path):
    videos_dir, specs = synth_videos
    for fmt in ["npy", "png"]:
        cache = tmp_path / fmt
        idx = frame_cache.build(videos_dir, str(cache), fmt=fmt, workers=2)
        assert idx["n_errors"] == 0 and idx["n_videos"] == len(specs)
        for vid in specs:
            ref, ref_idx = VideoCapture.load_frames_from_video(os.path.join(videos_dir, vid + ".mp4"), 12, "uniform")
            ours = torch.from_numpy(frame_cache.load_frames(str(cache), vid)).permute(0, 3, 1, 2).float() / 255
            assert ref.shape == ours.shape == (12, 3, 240, 320)
            assert torch.equal(ref, ours), f"{fmt}:{vid} differs"
            assert idx["videos"][vid]["frame_idxs"] == [int(i) for i in ref_idx]


def test_short_video_is_padded_like_official(synth_videos, tmp_path):
    videos_dir, _ = synth_videos
    arr, meta = frame_cache.decode_uint8(os.path.join(videos_dir, "video3.mp4"))  # 9 frames < 12
    assert arr.shape[0] == 12 and len(meta["frame_idxs"]) == 9
    assert (arr[9:] == arr[8]).all()
