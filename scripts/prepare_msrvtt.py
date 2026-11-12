"""Download MSR-VTT raw videos once, keep a copy on Google Drive, extract to fast local disk.

Sources (tried in order):
  1. Frozen-in-Time mirror used by CLIP4Clip / X-Pool README:
     https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip   (MSRVTT/videos/all/*.mp4)
  2. Hugging Face mirror: datasets/zhanjun/MSR-VTT  file MSR-VTT.tar          (MSR-VTT/all/*.mp4)

Annotations/splits are NOT downloaded: the official X-Pool repo ships them in
third_party/xpool/data/MSRVTT (MSRVTT_data.json, MSRVTT_train.9k.csv, MSRVTT_JSFUSION_test.csv).

Usage (Colab):
  python scripts/prepare_msrvtt.py --drive_dir /content/drive/MyDrive/cv_hw2/data \
      --local_dir /content/data/msrvtt
Writes <local_dir>/videos/videoN.mp4 (flat) and <local_dir>/prepare_report.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import tarfile
import time
import zipfile

OX_URL = "https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
HF_REPO, HF_FILE = "zhanjun/MSR-VTT", "MSR-VTT.tar"


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.call(cmd, shell=True)


def fetch_archive(drive_dir, local_tmp):
    """Return path of a local archive; reuse the Drive copy if it exists."""
    os.makedirs(local_tmp, exist_ok=True)
    for name in ["MSRVTT.zip", "MSR-VTT.tar"]:
        cached = os.path.join(drive_dir, name)
        if os.path.exists(cached) and os.path.getsize(cached) > 1e9:
            dst = os.path.join(local_tmp, name)
            if not os.path.exists(dst):
                t = time.time()
                shutil.copy(cached, dst)
                print(f"copied {name} from Drive in {time.time() - t:.0f}s")
            return dst, "drive-cache"
    dst = os.path.join(local_tmp, "MSRVTT.zip")
    t = time.time()
    if sh(f"wget -q --show-progress -O {dst} {OX_URL}") == 0 and os.path.getsize(dst) > 1e9:
        src = "robots.ox.ac.uk"
    else:
        if os.path.exists(dst):
            os.remove(dst)
        from huggingface_hub import hf_hub_download
        dst = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", local_dir=local_tmp)
        src = f"hf:{HF_REPO}"
    print(f"downloaded from {src} in {time.time() - t:.0f}s")
    os.makedirs(drive_dir, exist_ok=True)
    t = time.time()
    shutil.copy(dst, os.path.join(drive_dir, os.path.basename(dst)))
    print(f"saved copy to Drive in {time.time() - t:.0f}s")
    return dst, src


def extract_flat(archive, out_videos):
    os.makedirs(out_videos, exist_ok=True)
    n = 0
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.filename.endswith(".mp4"):
                    with z.open(m) as src, open(os.path.join(out_videos, os.path.basename(m.filename)), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n += 1
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".mp4"):
                    f = t.extractfile(m)
                    with open(os.path.join(out_videos, os.path.basename(m.name)), "wb") as dst:
                        shutil.copyfileobj(f, dst)
                    n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drive_dir", required=True)
    p.add_argument("--local_dir", default="/content/data/msrvtt")
    a = p.parse_args()
    t0 = time.time()
    archive, source = fetch_archive(a.drive_dir, os.path.join(a.local_dir, "_archive"))
    t1 = time.time()
    videos = os.path.join(a.local_dir, "videos")
    n = extract_flat(archive, videos)
    t2 = time.time()
    ids = {f[:-4] for f in os.listdir(videos) if f.endswith(".mp4")}
    expected = {f"video{i}" for i in range(10000)}
    report = {"source": source, "archive": os.path.basename(archive),
              "archive_GB": round(os.path.getsize(archive) / 1e9, 2),
              "n_mp4_extracted": n, "n_unique_ids": len(ids),
              "missing_of_10000": sorted(expected - ids)[:50], "n_missing": len(expected - ids),
              "fetch_seconds": round(t1 - t0, 1), "extract_seconds": round(t2 - t1, 1),
              "videos_GB": round(sum(os.path.getsize(os.path.join(videos, f)) for f in os.listdir(videos)) / 1e9, 2)}
    json.dump(report, open(os.path.join(a.local_dir, "prepare_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    shutil.rmtree(os.path.join(a.local_dir, "_archive"), ignore_errors=True)


if __name__ == "__main__":
    main()
