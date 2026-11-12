"""Run bookkeeping required by the assignment's Dataset Criterion:
GPU model + memory, batch size, epochs, peak memory usage and total training time
are written to <run_dir>/run_meta.json for every run, plus per-eval metrics to
<run_dir>/eval_history.jsonl. Nothing here touches the model.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import torch


def _git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty(path):
    try:
        out = subprocess.check_output(["git", "-C", path, "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:  # noqa: BLE001
        return None


def environment_info(repo_root=None):
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gpu_total_memory_GB"] = round(p.total_memory / 1024 ** 3, 2)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    try:
        import psutil
        info["ram_total_GB"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    du = shutil.disk_usage("/")
    info["disk_free_GB"] = round(du.free / 1024 ** 3, 1)
    if repo_root:
        info["repo_commit"] = _git_rev(repo_root)
        info["repo_dirty"] = _git_dirty(repo_root)
    return info


class RunLogger:
    def __init__(self, run_dir, config, repo_root=None, xpool_commit=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.t_start = time.time()
        self.meta = {
            "exp_name": getattr(config, "exp_name", None),
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "env": environment_info(repo_root),
            "xpool_commit": xpool_commit,
            "config": {k: v for k, v in vars(config).items() if _jsonable(v)},
        }
        self.hist_path = os.path.join(run_dir, "eval_history.jsonl")
        self.flush()

    def log_eval(self, record):
        record = {k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (str, bool)) else v)
                  for k, v in record.items()}
        record["wall_seconds"] = round(time.time() - self.t_start, 1)
        with open(self.hist_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def update(self, **kw):
        self.meta.update(kw)
        self.flush()

    def finish(self, **kw):
        self.meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.meta["total_wall_hours"] = round((time.time() - self.t_start) / 3600, 3)
        if torch.cuda.is_available():
            self.meta["peak_gpu_mem_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            self.meta["peak_gpu_mem_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        self.meta.update(kw)
        self.flush()

    def flush(self):
        with open(os.path.join(self.run_dir, "run_meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2, default=str)


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except TypeError:
        return False


class TimedLoader:
    """Wraps a DataLoader; measures time spent waiting for data vs. total step time."""

    def __init__(self, loader, max_steps=-1):
        self.loader = loader
        self.max_steps = max_steps
        self.data_wait = []
        self.step_time = []

    def __len__(self):
        n = len(self.loader)
        return n if self.max_steps is None or self.max_steps < 0 else min(n, self.max_steps)

    def __iter__(self):
        it = iter(self.loader)
        n = len(self)
        last = time.time()
        for _ in range(n):
            t0 = time.time()
            batch = next(it)
            t1 = time.time()
            self.data_wait.append(t1 - t0)
            if len(self.data_wait) > 1:
                self.step_time.append(t0 - last)
            last = t0
            yield batch

    def summary(self, skip=10):
        import numpy as np
        dw = np.array(self.data_wait[skip:]) if len(self.data_wait) > skip else np.array(self.data_wait)
        st = np.array(self.step_time[skip:]) if len(self.step_time) > skip else np.array(self.step_time)
        if len(st) == 0:
            return {}
        return {"steps_measured": int(len(st)), "sec_per_step_mean": float(st.mean()),
                "sec_per_step_median": float(np.median(st)), "data_wait_mean": float(dw.mean()),
                "data_wait_fraction": float(dw.mean() / max(st.mean(), 1e-9))}
