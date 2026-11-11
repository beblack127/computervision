"""Argument parsing that mirrors the official X-Pool AllConfig *exactly* (same names,
same defaults), plus a few clearly separated extra options for this project.

Official defaults are reproduced from third_party/xpool/config/all_config.py
(commit bc88718). If you change anything in the "official" block, you are no
longer running the baseline -- add a new option in the "project" block instead.
"""
import argparse
import os
from types import SimpleNamespace

from tvr.xpool_bridge import XPOOL_DATA


def add_official_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("official X-Pool arguments (defaults identical to all_config.py)")
    # data
    g.add_argument("--dataset_name", type=str, default="MSRVTT")
    g.add_argument("--videos_dir", type=str, default="data/MSRVTT/vids",
                   help="Raw .mp4 directory (only used when --cache_dir is not given)")
    g.add_argument("--msrvtt_train_file", type=str, default="9k")
    g.add_argument("--num_frames", type=int, default=12)
    g.add_argument("--video_sample_type", default="uniform", help="'rand'/'uniform'")
    g.add_argument("--input_res", type=int, default=224)
    # experiment
    g.add_argument("--exp_name", type=str, required=True)
    g.add_argument("--output_dir", type=str, default="./outputs")
    g.add_argument("--save_every", type=int, default=1)
    g.add_argument("--log_step", type=int, default=10)
    g.add_argument("--evals_per_epoch", type=int, default=10)
    g.add_argument("--load_epoch", type=int)
    g.add_argument("--eval_window_size", type=int, default=5)
    g.add_argument("--metric", type=str, default="t2v")
    # model
    g.add_argument("--huggingface", action="store_true", default=False)
    g.add_argument("--arch", type=str, default="clip_transformer")
    g.add_argument("--clip_arch", type=str, default="ViT-B/32")
    g.add_argument("--embed_dim", type=int, default=512)
    # training
    g.add_argument("--loss", type=str, default="clip")
    g.add_argument("--clip_lr", type=float, default=1e-6)
    g.add_argument("--noclip_lr", type=float, default=1e-5)
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--num_epochs", type=int, default=5)
    g.add_argument("--weight_decay", type=float, default=0.2)
    g.add_argument("--warmup_proportion", type=float, default=0.1)
    # pooling
    g.add_argument("--pooling_type", type=str)
    g.add_argument("--k", type=int, default=-1)
    g.add_argument("--attention_temperature", type=float, default=0.01)
    g.add_argument("--num_mha_heads", type=int, default=1)
    g.add_argument("--transformer_dropout", type=float, default=0.3)
    # system
    g.add_argument("--num_workers", type=int, default=8)
    g.add_argument("--seed", type=int, default=24)
    g.add_argument("--no_tensorboard", action="store_true", default=False)
    g.add_argument("--tb_log_dir", type=str, default="logs")


def add_project_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("project-only arguments (do not change the baseline math)")
    g.add_argument("--data_dir", type=str, default=XPOOL_DATA,
                   help="Directory with MSRVTT_data.json / *_train.*.csv / JSFUSION_test.csv")
    g.add_argument("--cache_dir", type=str, default=None,
                   help="Frame cache built by tvr.frame_cache (bit-identical to on-the-fly decoding)")
    g.add_argument("--max_steps", type=int, default=-1,
                   help="Stop after this many train steps (benchmarking only; -1 = full schedule)")
    g.add_argument("--save_optimizer", action="store_true", default=False,
                   help="Also store optimizer state in model_best.pth (official code does; ~3x larger)")
    g.add_argument("--save_last", action="store_true", default=True,
                   help="Store last-step weights as model_last.pth (for 'last checkpoint' reporting)")
    g.add_argument("--resume_state", type=str, default=None,
                   help="Path to resume_state.pth written at the end of an epoch (Colab disconnects)")
    g.add_argument("--notes", type=str, default="", help="Free text stored in run_meta.json")


def build_parser(description="X-Pool (project wrapper)"):
    p = argparse.ArgumentParser(description=description)
    add_official_args(p)
    add_project_args(p)
    return p


def finalize(args):
    """Same post-processing as AllConfig.parse_args (paths), without deleting anything."""
    args.model_path = os.path.join(args.output_dir, args.exp_name)
    args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)
    return SimpleNamespace(**vars(args))
