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


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _previous_sessions(prev):
    """Sessions recorded by an earlier (interrupted or finished) run of the same experiment."""
    if not prev:
        return []
    if "sessions" in prev:
        return list(prev["sessions"])
    env = prev.get("env", {})  # run_meta.json written before session tracking existed
    return [{"index": 0, "started_at": prev.get("started_at"), "last_update": prev.get("finished_at"),
             "finished_at": prev.get("finished_at"), "wall_hours": prev.get("total_wall_hours"),
             "train_seconds": prev.get("train_seconds"), "gpu_name": env.get("gpu_name"),
             "gpu_total_memory_GB": env.get("gpu_total_memory_GB"),
             "peak_gpu_mem_reserved_GB": prev.get("peak_gpu_mem_reserved_GB"),
             "note": "legacy record: sessions before this one were not tracked"}]


class RunLogger:
    """Writes run_meta.json and eval_history.jsonl.

    A run interrupted on Colab is continued with --resume_state. Each process is one *session*;
    run_meta.json keeps every session (time, GPU, peak memory) so totals cover the whole run.
    Every eval record carries its session index; `wall_seconds` is measured from session start.
    """

    def __init__(self, run_dir, config, repo_root=None, xpool_commit=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.t_start = time.time()
        self.meta_path = os.path.join(run_dir, "run_meta.json")
        prev = None
        if getattr(config, "resume_state", None) and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path) as f:
                    prev = json.load(f)
            except (OSError, ValueError):
                prev = None
        env = environment_info(repo_root)
        self.sessions = _previous_sessions(prev)
        self.session = {"index": len(self.sessions), "started_at": _now(), "last_update": None,
                        "wall_hours": 0.0, "gpu_name": env.get("gpu_name"),
                        "gpu_total_memory_GB": env.get("gpu_total_memory_GB"),
                        "resumed": bool(getattr(config, "resume_state", None))}
        self.sessions.append(self.session)
        self.meta = {
            "exp_name": getattr(config, "exp_name", None),
            "started_at": self.sessions[0].get("started_at") or self.session["started_at"],
            "env": env,
            "xpool_commit": xpool_commit,
            "config": {k: v for k, v in vars(config).items() if _jsonable(v)},
        }
        if prev and "best_checkpoint" in prev:
            self.meta["best_checkpoint"] = prev["best_checkpoint"]
        self.hist_path = os.path.join(run_dir, "eval_history.jsonl")
        self.flush()

    def log_eval(self, record):
        record = {k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (str, bool)) else v)
                  for k, v in record.items()}
        record["session"] = self.session["index"]
        record["time"] = _now()
        record["wall_seconds"] = round(time.time() - self.t_start, 1)
        with open(self.hist_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.flush()  # heartbeat: an interrupted session still records its time up to the last eval

    def update(self, **kw):
        self.meta.update(kw)
        self.flush()

    def finish(self, **kw):
        self.session["finished_at"] = _now()
        if torch.cuda.is_available():
            self.session["peak_gpu_mem_allocated_GB"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
            self.session["peak_gpu_mem_reserved_GB"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
            self.meta["peak_gpu_mem_allocated_GB"] = self.session["peak_gpu_mem_allocated_GB"]
            self.meta["peak_gpu_mem_reserved_GB"] = self.session["peak_gpu_mem_reserved_GB"]
        if "train_seconds" in kw:
            self.session["train_seconds"] = kw["train_seconds"]
        self.meta["finished_at"] = self.session["finished_at"]
        self.meta.update(kw)
        self.flush()

    def flush(self):
        self.session["last_update"] = _now()
        self.session["wall_hours"] = round((time.time() - self.t_start) / 3600, 3)
        self.meta["sessions"] = self.sessions
        self.meta["n_sessions"] = len(self.sessions)
        self.meta["this_session_wall_hours"] = self.session["wall_hours"]
        self.meta["total_wall_hours"] = round(sum(s.get("wall_hours") or 0 for s in self.sessions), 3)
        peaks = [s["peak_gpu_mem_reserved_GB"] for s in self.sessions if s.get("peak_gpu_mem_reserved_GB")]
        if peaks:
            self.meta["peak_gpu_mem_reserved_GB_all_sessions"] = max(peaks)
        self.meta["gpus_used"] = sorted({s["gpu_name"] for s in self.sessions if s.get("gpu_name")})
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        os.replace(tmp, self.meta_path)


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
