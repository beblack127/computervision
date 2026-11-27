"""Embed a querybank of TRAINING captions with a trained X-Pool checkpoint (for QB-Norm).

QB-Norm (Bogolin et al., CVPR 2022) normalises test similarities with the similarities of a bank of
training queries to the gallery; it needs no test queries. Defaults follow the paper's ablations:
5,000 captions sampled from the training split.

  python analysis/embed_querybank.py --ckpt RUN/model_best.pth --data_dir third_party/xpool/data/MSRVTT \
      --out RUN/querybank_9k_5000.pt
"""
import argparse
import csv
import json
import os
import random
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def sample_captions(data_dir, train_csv="MSRVTT_train.9k.csv", test_csv="MSRVTT_JSFUSION_test.csv", n=5000, seed=0):
    """n captions of training videos (1k-A test videos excluded), reproducible."""
    with open(os.path.join(data_dir, train_csv)) as f:
        train = {r["video_id"] for r in csv.DictReader(f)}
    test = set()
    p = os.path.join(data_dir, test_csv)
    if os.path.exists(p):
        with open(p) as f:
            test = {r["video_id"] for r in csv.DictReader(f)}
    assert not train & test, "train/test overlap"
    db = json.load(open(os.path.join(data_dir, "MSRVTT_data.json")))
    pool = sorted(((s["video_id"], s["caption"]) for s in db["sentences"] if s["video_id"] in train))
    rng = random.Random(seed)
    picked = rng.sample(pool, min(n, len(pool)))
    return [c for _, c in picked], [v for v, _ in picked]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    from tvr.compat import build_tokenizer, load_xpool_state_dict, safe_torch_load
    from tvr.xpool_bridge import ensure_xpool_on_path
    ensure_xpool_on_path()
    from model.clip_transformer import CLIPTransformer  # official

    texts, vids = sample_captions(a.data_dir, n=a.n, seed=a.seed)
    cfg = SimpleNamespace(huggingface=True, clip_arch="ViT-B/32", embed_dim=512, num_mha_heads=1,
                          transformer_dropout=0.3, input_res=224, num_frames=12, pooling_type=None)
    model = CLIPTransformer(cfg)
    info = load_xpool_state_dict(model, safe_torch_load(a.ckpt))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    tok = build_tokenizer(True)
    embs = []
    for i in range(0, len(texts), a.batch):
        t = tok(texts[i:i + a.batch], return_tensors="pt", padding=True, truncation=True)
        embs.append(model.clip.get_text_features(**{k: v.to(dev) for k, v in t.items()}).float().cpu())
    E = torch.cat(embs)
    torch.save({"text_embeds": E, "texts": texts, "video_ids": vids, "n": len(texts), "seed": a.seed,
                "ckpt": a.ckpt, "load_info": info}, a.out)
    print(f"[querybank] {E.shape[0]} training captions from {len(set(vids))} videos -> {a.out}")


if __name__ == "__main__":
    main()
