"""Frame cache that is bit-identical to the official on-the-fly decoding.

Why this is allowed: the official config uses video_sample_type='uniform' for BOTH
train and test (third_party/xpool/config/all_config.py), so every epoch reads the
same 12 frames per video. Decoding them once and storing the raw uint8 RGB frames
gives exactly the tensors VideoCapture.load_frames_from_video() would return
(verified by tests/test_frame_cache.py and, on real data, by `verify` below).
Random augmentations (RandomResizedCrop / flip) are still applied on the fly.

CLI:
  python -m tvr.frame_cache build  --videos_dir V --cache_dir C [--format npy|png] [--workers N]
  python -m tvr.frame_cache verify --videos_dir V --cache_dir C [--n 50]
  python -m tvr.frame_cache stats  --cache_dir C
"""
import argparse
import json
import os
import random
import time
from multiprocessing import Pool

import cv2
import numpy as np

NUM_FRAMES = 12


def uniform_indices(vlen, num_frames=NUM_FRAMES, sample="uniform"):
    """Exact copy of the index logic in xpool/datasets/video_capture.py."""
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = [(interv, intervals[i + 1] - 1) for i, interv in enumerate(intervals[:-1])]
    if sample == "rand":
        return [random.choice(range(x[0], x[1])) for x in ranges]
    return [(x[0] + x[1]) // 2 for x in ranges]


def decode_uint8(video_path, num_frames=NUM_FRAMES, sample="uniform"):
    """Mirror of VideoCapture.load_frames_from_video but returns uint8 RGB (T,H,W,3) + meta.

    The official function returns torch.stack(frames).float()/255 with frames in (C,H,W);
    torch.from_numpy(out).permute(0,3,1,2).float()/255 reproduces it exactly.
    """
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), video_path
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_idxs = uniform_indices(vlen, num_frames, sample)
    frames = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret:
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    break
        if not ret:
            cap.release()
            raise ValueError(f"cannot read frame {index} of {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    cap.release()
    arr = np.stack(frames).astype(np.uint8)
    meta = {"vlen": vlen, "fps": fps, "frame_idxs": [int(i) for i in frame_idxs],
            "shape": list(arr.shape)}
    return arr, meta


def _cache_path(cache_dir, vid, fmt):
    return os.path.join(cache_dir, vid + (".npy" if fmt == "npy" else ""))


def save_frames(cache_dir, vid, arr, fmt="npy"):
    if fmt == "npy":
        np.save(_cache_path(cache_dir, vid, fmt), arr)
    else:  # lossless PNG per frame, ~2x smaller, slower to read
        d = _cache_path(cache_dir, vid, fmt)
        os.makedirs(d, exist_ok=True)
        for i, fr in enumerate(arr):
            cv2.imwrite(os.path.join(d, f"{i:02d}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))


def load_frames(cache_dir, vid):
    """Return uint8 (T,H,W,3) RGB from whichever format is present."""
    p = os.path.join(cache_dir, vid + ".npy")
    if os.path.exists(p):
        return np.load(p)
    d = os.path.join(cache_dir, vid)
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    return np.stack([cv2.cvtColor(cv2.imread(os.path.join(d, f)), cv2.COLOR_BGR2RGB) for f in files])


def _work(job):
    video_path, cache_dir, fmt = job
    vid = os.path.splitext(os.path.basename(video_path))[0]
    try:
        cv2.setNumThreads(1)
        arr, meta = decode_uint8(video_path)
        save_frames(cache_dir, vid, arr, fmt)
        meta["video_id"] = vid
        meta["bytes"] = int(arr.nbytes)
        return meta
    except Exception as e:  # noqa: BLE001
        return {"video_id": vid, "error": repr(e)}


def build(videos_dir, cache_dir, fmt="npy", workers=None, video_ids=None):
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(videos_dir) if f.endswith(".mp4"))
    if video_ids is not None:
        keep = set(video_ids)
        files = [f for f in files if f[:-4] in keep]
    jobs = [(os.path.join(videos_dir, f), cache_dir, fmt) for f in files]
    workers = workers or os.cpu_count() or 4
    t0 = time.time()
    metas = []
    with Pool(workers) as pool:
        for i, m in enumerate(pool.imap_unordered(_work, jobs, chunksize=8)):
            metas.append(m)
            if (i + 1) % 1000 == 0:
                print(f"[frame_cache] {i + 1}/{len(jobs)} videos, {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0
    errors = [m for m in metas if "error" in m]
    index = {"format": fmt, "num_frames": NUM_FRAMES, "sample": "uniform",
             "n_videos": len(metas) - len(errors), "n_errors": len(errors),
             "build_seconds": round(elapsed, 1), "workers": workers,
             "total_bytes_raw": int(sum(m.get("bytes", 0) for m in metas)),
             "videos": {m["video_id"]: m for m in metas if "error" not in m},
             "errors": errors}
    with open(os.path.join(cache_dir, "index.json"), "w") as f:
        json.dump(index, f)
    print(f"[frame_cache] done: {index['n_videos']} ok, {len(errors)} errors, {elapsed:.0f}s")
    return index


def verify(videos_dir, cache_dir, n=50, seed=0):
    """Compare cache against the *official* VideoCapture for n random videos (bitwise)."""
    import torch
    from tvr.xpool_bridge import ensure_xpool_on_path
    ensure_xpool_on_path()
    from datasets.video_capture import VideoCapture  # official
    idx = json.load(open(os.path.join(cache_dir, "index.json")))
    vids = sorted(idx["videos"].keys())
    random.Random(seed).shuffle(vids)
    bad = []
    for vid in vids[:n]:
        ref, _ = VideoCapture.load_frames_from_video(os.path.join(videos_dir, vid + ".mp4"), NUM_FRAMES, "uniform")
        ours = torch.from_numpy(load_frames(cache_dir, vid)).permute(0, 3, 1, 2).float() / 255
        if ref.shape != ours.shape or not torch.equal(ref, ours):
            bad.append(vid)
    print(f"[frame_cache] verify: {min(n, len(vids)) - len(bad)}/{min(n, len(vids))} bit-identical")
    return {"checked": min(n, len(vids)), "mismatched": bad}


def stats(cache_dir):
    idx = json.load(open(os.path.join(cache_dir, "index.json")))
    shapes = {}
    vlens, fpss = [], []
    for m in idx["videos"].values():
        hw = f"{m['shape'][1]}x{m['shape'][2]}"
        shapes[hw] = shapes.get(hw, 0) + 1
        vlens.append(m["vlen"])
        fpss.append(m["fps"])
    out = {"n_videos": idx["n_videos"], "n_errors": idx["n_errors"],
           "build_seconds": idx["build_seconds"], "total_GB_raw": idx["total_bytes_raw"] / 1e9,
           "resolutions_top": sorted(shapes.items(), key=lambda x: -x[1])[:10],
           "frames_per_video": {"min": int(np.min(vlens)), "median": float(np.median(vlens)), "max": int(np.max(vlens))},
           "fps": {"min": float(np.min(fpss)), "median": float(np.median(fpss)), "max": float(np.max(fpss))}}
    print(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "verify", "stats"])
    p.add_argument("--videos_dir")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--format", default="npy", choices=["npy", "png"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--n", type=int, default=50)
    a = p.parse_args()
    if a.cmd == "build":
        build(a.videos_dir, a.cache_dir, a.format, a.workers)
    elif a.cmd == "verify":
        verify(a.videos_dir, a.cache_dir, a.n)
    else:
        stats(a.cache_dir)


if __name__ == "__main__":
    main()
