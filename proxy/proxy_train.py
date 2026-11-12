"""Train only the pooling head on frozen features, evaluate on MSR-VTT 1k-A (minutes per run).

PURPOSE: cheap *screening* of ideas before spending A100 hours on full end-to-end runs.
Proxy numbers are NOT comparable to the paper and must never be reported as the final result;
they only rank ideas against the proxy baseline ('xpool' head) under identical conditions.

  python proxy/proxy_train.py --feats FEATS/xpool_s24_feats.pt --head xpool --seed 0 --out OUT.json
  python proxy/proxy_train.py --feats ... --head xpool_tpe   # example plug-in (temporal positions)

Add a new idea = add a class to HEADS with forward(text [T,D], frames [V,F,D]) -> [V,T,D] or [V,D].
"""
import argparse
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tvr.xpool_bridge import ensure_xpool_on_path  # noqa: E402

ensure_xpool_on_path()
from modules.loss import CLIPLoss  # noqa: E402  (official)
from modules.metrics import t2v_metrics, v2t_metrics  # noqa: E402
from modules.transformer import Transformer  # noqa: E402


def xp_cfg(dropout=0.3):
    return SimpleNamespace(embed_dim=512, num_mha_heads=1, transformer_dropout=dropout)


class MeanHead(nn.Module):
    text_conditioned = False

    def forward(self, text, frames):
        return frames.mean(1)


class XPoolHead(nn.Module):
    """Official X-Pool pooling (third_party/xpool/modules/transformer.py)."""
    text_conditioned = True

    def __init__(self, dropout=0.3):
        super().__init__()
        self.pool = Transformer(xp_cfg(dropout))

    def forward(self, text, frames):
        return self.pool(text, frames)


class XPoolTPEHead(XPoolHead):
    """EXAMPLE plug-in: add learnable temporal position embeddings to frames before X-Pool.
    Tests the hypothesis 'X-Pool is order-invariant, so order-dependent queries suffer'."""

    def __init__(self, dropout=0.3, num_frames=12):
        super().__init__(dropout)
        self.tpe = nn.Parameter(torch.zeros(1, num_frames, 512))

    def forward(self, text, frames):
        return self.pool(text, frames + self.tpe)


HEADS = {"mean": MeanHead, "xpool": XPoolHead, "xpool_tpe": XPoolTPEHead}


def sims_for(head, text, frames, chunk=256):
    """text [T,D], frames [V,F,D] -> sims [T,V] (cosine), chunked over videos."""
    t = F.normalize(text, dim=-1)
    if not head.text_conditioned:
        return t @ F.normalize(head(text, frames), dim=-1).t()
    out = []
    for i in range(0, frames.shape[0], chunk):
        pooled = F.normalize(head(text, frames[i:i + chunk]), dim=-1)  # [v, T, D]
        out.append(torch.einsum("td,vtd->tv", t, pooled))
    return torch.cat(out, dim=1)


@torch.no_grad()
def evaluate(head, feats, dev):
    head.eval()
    vid_index = {v: i for i, v in enumerate(feats["video_ids"])}
    vi = torch.tensor([vid_index[v] for v in feats["test_video_ids"]])
    frames = feats["vid_feats"][vi].float().to(dev)
    text = feats["test_feats"].float().to(dev)
    sims = sims_for(head, text, frames).cpu()  # [query i, video j], GT = i
    s3 = sims[:, None, :]  # official layout: num_vids x max_text_per_vid x num_vids
    return {"t2v": t2v_metrics(s3.clone()), "v2t": v2t_metrics(s3.clone())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feats", required=True)
    p.add_argument("--head", default="xpool", choices=sorted(HEADS))
    p.add_argument("--train_file", default="9k")
    p.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "..", "third_party", "xpool",
                                                   "data", "MSRVTT"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--wd", type=float, default=0.2)
    p.add_argument("--train_logit_scale", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = torch.load(a.feats, weights_only=False)
    import pandas as pd
    train_vids = set(pd.read_csv(os.path.join(a.data_dir, f"MSRVTT_train.{a.train_file}.csv"))["video_id"])
    keep = torch.tensor([feats["video_ids"][int(v)] in train_vids for v in feats["cap_vid_idx"]])
    cap_feats = feats["cap_feats"][keep].float()
    cap_vid = feats["cap_vid_idx"][keep]
    vid_feats = feats["vid_feats"].float()

    head = HEADS[a.head]().to(dev)
    logit_scale = nn.Parameter(torch.tensor(math.log(feats.get("logit_scale", 100.0))), requires_grad=a.train_logit_scale)
    params = [p_ for p_ in head.parameters()] + ([logit_scale] if a.train_logit_scale else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd) if params else None
    loss_fn = CLIPLoss()
    n = len(cap_vid)
    steps = a.epochs * (n // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, 0.1 * steps)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))) if opt else None
    t0 = time.time()
    hist = []
    for ep in range(a.epochs if opt else 0):
        head.train()
        perm = torch.randperm(n)
        for s in range(n // a.bs):
            idx = perm[s * a.bs:(s + 1) * a.bs]
            text = cap_feats[idx].to(dev)
            frames = vid_feats[cap_vid[idx]].to(dev)
            pooled = head(text, frames)
            t = F.normalize(text, dim=-1)
            if head.text_conditioned:  # official sim_matrix_training layout
                pooled = F.normalize(pooled, dim=-1).permute(1, 2, 0)
                sims = torch.bmm(t.unsqueeze(1), pooled).squeeze(1)
            else:
                sims = t @ F.normalize(pooled, dim=-1).t()
            loss = loss_fn(sims, logit_scale.to(dev))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        res = evaluate(head, feats, dev)
        hist.append({"epoch": ep + 1, **{f"t2v_{k}": v for k, v in res["t2v"].items()}})
        print(f"epoch {ep + 1}: t2v R1 {res['t2v']['R1']:.1f}  R5 {res['t2v']['R5']:.1f}  MnR {res['t2v']['MeanR']:.1f}",
              flush=True)
    res = evaluate(head, feats, dev)
    out = {"head": a.head, "seed": a.seed, "feats": a.feats, "epochs": a.epochs, "bs": a.bs, "lr": a.lr,
           "seconds": round(time.time() - t0, 1), "final": res, "history": hist,
           "note": "PROXY (frozen features). Use only to rank ideas against the 'xpool' proxy baseline."}
    print(json.dumps({"head": a.head, "t2v": res["t2v"]}, indent=2, default=float))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
