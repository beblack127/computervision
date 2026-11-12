"""Qualitative failure gallery with X-Pool's text->frame attention weights.

For selected failed queries we show the 12 frames of (a) the ground-truth video and
(b) the wrongly retrieved top-1 video, each frame annotated with the attention weight
the query assigns to it inside X-Pool's cross-modal pooling. Runs on CPU.

  python analysis/failure_gallery.py --dump RUN/dump_model_best/dump_original.pt \
      --ckpt RUN/model_best.pth --cache_dir /content/cache/msrvtt_u12 --out_dir RUN/analysis/gallery
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tvr.compat import safe_torch_load  # noqa: E402
from tvr.frame_cache import load_frames  # noqa: E402
from tvr.xpool_bridge import ensure_xpool_on_path  # noqa: E402

ensure_xpool_on_path()
from modules.transformer import Transformer  # noqa: E402  (official pooling module)


def load_pooling(ckpt_path, num_heads=1, dropout=0.3):
    sd = safe_torch_load(ckpt_path)
    sd = sd.get("state_dict", sd)
    pool_sd = {k[len("pool_frames."):]: v for k, v in sd.items() if k.startswith("pool_frames.")}
    m = Transformer(SimpleNamespace(embed_dim=512, num_mha_heads=num_heads, transformer_dropout=dropout))
    m.load_state_dict(pool_sd, strict=True)
    return m.eval()


@torch.no_grad()
def frame_attention(pool, text_embed, frame_embeds):
    """Attention of one text over the frames of one video (heads averaged). Mirrors
    MultiHeadedAttention.forward in third_party/xpool/modules/transformer.py."""
    t = pool.layer_norm1(text_embed[None])
    v = pool.layer_norm1(frame_embeds[None])
    attn = pool.cross_attn
    q = attn.q_proj(t).reshape(1, attn.num_heads, attn.head_dim).permute(1, 2, 0)
    k = attn.k_proj(v).reshape(1, v.shape[1], attn.num_heads, attn.head_dim).permute(0, 2, 1, 3)
    logits = (k @ q) / math.sqrt(attn.head_dim)  # 1 x heads x frames x 1
    w = torch.softmax(logits, dim=2)[0, :, :, 0]  # heads x frames
    return w.mean(0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--select", default="worst", choices=["worst", "near_miss", "random"])
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out_dir, exist_ok=True)
    d = torch.load(a.dump, weights_only=False)
    sims, texts, vids = d["sims"].float(), d["texts"], d["video_ids"]
    te, ve = d["text_embeds"].float(), d["vid_embeds"].float()
    order = torch.argsort(sims, dim=1, descending=True).numpy()
    ranks = (order == np.arange(len(vids))[:, None]).argmax(1)
    fails = np.where(ranks > 0)[0]
    if a.select == "worst":
        pick = fails[np.argsort(-ranks[fails])][: a.n]
    elif a.select == "near_miss":
        pick = fails[np.argsort(ranks[fails])][: a.n]
    else:
        pick = np.random.default_rng(0).choice(fails, size=min(a.n, len(fails)), replace=False)
    pool = load_pooling(a.ckpt)
    md = ["# Failure gallery", ""]
    for qi in pick:
        wrong = order[qi, 0]
        rows = [("GT " + vids[qi], qi), ("top-1 " + vids[wrong], wrong)]
        fig, axes = plt.subplots(2, 12, figsize=(18, 3.6))
        for r, (label, vi) in enumerate(rows):
            frames = load_frames(a.cache_dir, vids[vi])
            w = frame_attention(pool, te[qi], ve[vi])
            for f in range(12):
                ax = axes[r, f]
                ax.imshow(frames[f]); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{w[f]:.2f}", fontsize=8, color="red" if w[f] == w.max() else "black")
            axes[r, 0].set_ylabel(label, fontsize=8)
        fig.suptitle(f"query: \"{texts[qi]}\"  |  GT rank {ranks[qi] + 1}  |  top-1 caption: \"{texts[wrong]}\"", fontsize=9)
        fig.tight_layout()
        name = f"q{qi:04d}_{vids[qi]}.png"
        fig.savefig(os.path.join(a.out_dir, name), dpi=110)
        plt.close(fig)
        md += [f"## {texts[qi]}", f"GT {vids[qi]} at rank {ranks[qi] + 1}; top-1 {vids[wrong]} (\"{texts[wrong]}\")",
               "", f"![]({name})", ""]
    open(os.path.join(a.out_dir, "index.md"), "w").write("\n".join(md))
    print(f"wrote {len(pick)} figures to {a.out_dir}")


if __name__ == "__main__":
    main()
