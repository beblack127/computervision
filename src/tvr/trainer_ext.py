"""Official X-Pool Trainer + bookkeeping (logging, lighter checkpoints, epoch-level resume).

The optimisation step, evaluation schedule (evals_per_epoch=10 on the 1k-A test set),
windowed model selection (eval_window_size=5) and metrics are inherited unchanged from
third_party/xpool/trainer/trainer.py.
"""
import os
import random

import numpy as np
import torch

from tvr.runlog import retry_io
from tvr.xpool_bridge import ensure_xpool_on_path

ensure_xpool_on_path()
from trainer.trainer import Trainer  # noqa: E402  (official)


class ProjectTrainer(Trainer):
    def __init__(self, *args, run_logger=None, save_optimizer=False, save_last=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_logger = run_logger
        self.save_optimizer = save_optimizer
        self.save_last = save_last
        self.current_epoch = 0
        self.best_record = None

    # ---- evaluation: identical computation, plus logging -------------------------------
    def _valid_epoch_step(self, epoch, step, num_steps):
        res = super()._valid_epoch_step(epoch, step, num_steps)
        if self.run_logger is not None:
            rec = {"epoch": epoch, "step": step, "global_step": self.global_step}
            rec.update({k: v for k, v in res.items()})
            self.run_logger.log_eval(rec)
        return res

    # ---- checkpoints: same selection rule, smaller files ---------------------------------
    def _save_checkpoint(self, epoch, save_best=False):
        if not save_best:
            return  # per-epoch full checkpoints are replaced by resume_state.pth (see train())
        state = {"epoch": epoch, "global_step": self.global_step, "state_dict": self.model.state_dict()}
        if self.save_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
        retry_io(lambda: torch.save(state, os.path.join(self.checkpoint_dir, "model_best.pth")), "save model_best.pth")
        self.best_record = {"epoch": epoch, "global_step": self.global_step,
                            "best_window_R1": float(self.best_window)}
        if self.run_logger is not None:
            self.run_logger.update(best_checkpoint=self.best_record)
        print("Saving current best: model_best.pth ...")

    def _resume_path(self):
        return os.path.join(self.checkpoint_dir, "resume_state.pth")

    def _save_resume_state(self, epoch):
        state = {
            "epoch": epoch, "global_step": self.global_step,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "best_window": self.best_window, "best": self.best,
            "window_metric": {k: list(v) for k, v in self.window_metric.items()},
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
            "best_record": self.best_record,
        }
        tmp = self._resume_path() + ".tmp"

        def _write():
            torch.save(state, tmp)
            os.replace(tmp, self._resume_path())

        retry_io(_write, "save resume_state.pth")

    def load_resume_state(self, path):
        s = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(s["model"])
        self.optimizer.load_state_dict(s["optimizer"])
        if self.lr_scheduler is not None and s["lr_scheduler"] is not None:
            self.lr_scheduler.load_state_dict(s["lr_scheduler"])
        self.global_step = s["global_step"]
        self.best_window, self.best = s["best_window"], s["best"]
        for k, v in s["window_metric"].items():
            self.window_metric[k].extend(v)
        torch.set_rng_state(s["rng"]["torch"])
        np.random.set_state(s["rng"]["numpy"])
        random.setstate(s["rng"]["python"])
        if s["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["rng"]["cuda"])
        self.best_record = s.get("best_record")
        self.start_epoch = s["epoch"] + 1
        print(f"[resume] continuing from epoch {self.start_epoch}, global_step {self.global_step}")

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            res = self._train_epoch(epoch)
            if self.run_logger is not None:
                self.run_logger.log_eval({"epoch": epoch, "type": "epoch_end", **res})
            self._save_resume_state(epoch)
        if self.save_last:
            last = {"epoch": self.num_epochs, "global_step": self.global_step, "state_dict": self.model.state_dict()}
            retry_io(lambda: torch.save(last, os.path.join(self.checkpoint_dir, "model_last.pth")), "save model_last.pth")
