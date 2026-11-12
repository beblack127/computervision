"""Train X-Pool on MSR-VTT with the official recipe (project wrapper).

Official command being reproduced (README of the authors, MSR-VTT-9k):
  python train.py --exp_name=... --videos_dir=... --batch_size=32 --noclip_lr=3e-5 \
      --transformer_dropout=0.3 --huggingface --dataset_name=MSRVTT --msrvtt_train_file=9k

Equivalent here:
  python -m tvr.train --exp_name xpool_9k_s24 --cache_dir /content/cache --batch_size 32 \
      --noclip_lr 3e-5 --transformer_dropout 0.3 --huggingface --dataset_name MSRVTT \
      --msrvtt_train_file 9k --output_dir "/content/drive/MyDrive/컴비 프로젝트/runs"

Benchmark mode (go/no-go): add --max_steps 200 --evals_per_epoch 0
"""
import json
import os
import random
import time

import numpy as np
import torch

from tvr.config import build_parser, finalize
from tvr.compat import build_tokenizer
from tvr.data import get_data_loader
from tvr.runlog import RunLogger, TimedLoader
from tvr.trainer_ext import ProjectTrainer
from tvr.xpool_bridge import REPO_ROOT, ensure_xpool_on_path, xpool_commit

ensure_xpool_on_path()
from model.model_factory import ModelFactory  # noqa: E402  (official)
from modules.loss import LossFactory  # noqa: E402
from modules.metrics import t2v_metrics, v2t_metrics  # noqa: E402
from modules.optimization import AdamW, get_cosine_schedule_with_warmup  # noqa: E402


def seed_everything(seed):
    # identical to the official train.py
    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_everything(config, for_training=True):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(config.seed)
    tokenizer = build_tokenizer(config.huggingface)
    train_loader = get_data_loader(config, "train") if for_training else None
    valid_loader = get_data_loader(config, "test")
    model = ModelFactory.get_model(config)
    metrics = t2v_metrics if config.metric == "t2v" else v2t_metrics
    loss = LossFactory.get_loss(config)
    return tokenizer, train_loader, valid_loader, model, metrics, loss


def main():
    args = build_parser("Train X-Pool (project wrapper)").parse_args()
    config = finalize(args)
    run_logger = RunLogger(config.model_path, config, REPO_ROOT, xpool_commit())
    tokenizer, train_loader, valid_loader, model, metrics, loss = build_everything(config)

    params = list(model.named_parameters())
    clip_params = [p for n, p in params if "clip." in n]
    noclip_params = [p for n, p in params if "clip." not in n]
    optimizer = AdamW([{"params": clip_params, "lr": config.clip_lr},
                       {"params": noclip_params, "lr": config.noclip_lr}],
                      weight_decay=config.weight_decay)
    num_training_steps = len(train_loader) * config.num_epochs  # full schedule, even when benchmarking
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    timed = TimedLoader(train_loader, config.max_steps)
    writer = None
    if not config.no_tensorboard:
        from torch.utils.tensorboard.writer import SummaryWriter
        writer = SummaryWriter(log_dir=config.tb_log_dir)

    trainer = ProjectTrainer(model, loss, metrics, optimizer, config=config,
                             train_data_loader=timed, valid_data_loader=valid_loader,
                             lr_scheduler=scheduler, writer=writer, tokenizer=tokenizer,
                             run_logger=run_logger, save_optimizer=config.save_optimizer,
                             save_last=config.save_last)
    run_logger.update(n_train_pairs=len(train_loader.dataset), steps_per_epoch=len(train_loader),
                      total_steps=num_training_steps, warmup_steps=num_warmup_steps,
                      n_test_pairs=len(valid_loader.dataset),
                      n_params_total=sum(p.numel() for p in model.parameters()),
                      n_params_pooling=sum(p.numel() for p in noclip_params))

    if config.resume_state:
        trainer.load_resume_state(config.resume_state)

    t0 = time.time()
    if config.max_steps > 0:  # benchmark: one truncated epoch, nothing saved but timings
        trainer._train_epoch(1)
        bench = timed.summary()
        bench["train_seconds"] = round(time.time() - t0, 1)
        if valid_loader is not None and config.evals_per_epoch == 0:
            te = time.time()
            trainer._valid_epoch_step(0, 0, 0)
            bench["one_eval_seconds"] = round(time.time() - te, 1)
        spe = len(train_loader)
        n_evals = 10  # official evals_per_epoch for the real run
        est = (bench["sec_per_step_median"] * spe * config.num_epochs
               + bench.get("one_eval_seconds", 0) * n_evals * config.num_epochs) / 3600
        bench["estimated_full_run_hours"] = round(est, 2)
        run_logger.finish(benchmark=bench)
        print(json.dumps(bench, indent=2))
        return

    trainer.train()
    run_logger.finish(train_seconds=round(time.time() - t0, 1),
                      sec_per_step=timed.summary())


if __name__ == "__main__":
    main()
