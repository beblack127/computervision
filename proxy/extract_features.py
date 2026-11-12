"""Extract frozen CLIP features for fast idea screening ("proxy" experiments).

Frame features use the deterministic test transform (no augmentation). With --ckpt they come
from the fine-tuned X-Pool CLIP (recommended: features of our reproduced baseline), without
it from zero-shot CLIP ViT-B/32.

  python proxy/extract_features.py --ckpt RUN/model_best.pth --cache_dir C \
      --data_dir third_party/xpool/data/MSRVTT --out FEATS/xpool_s24_feats.pt
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tvr.compat import build_tokenizer, init_transform_dict, load_xpool_state_dict, safe_torch_load  # noqa: E402
from tvr.frame_cache import load_frames  # noqa: E402
from tvr.xpool_bridge import ensure_xpool_on_path  # noqa: E402

ensure_xpool_on_path()
from model.clip_transformer import CLIPTransformer  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bs_videos", type=int, default=64)
    p.add_argument("--bs_text", type=int, default=1024)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimpleNamespace(huggingface=True, embed_dim=512, num_mha_heads=1, transformer_dropout=0.3,
                          input_res=224, num_frames=12)
    model = CLIPTransformer(cfg)
    if a.ckpt:
        load_xpool_state_dict(model, safe_torch_load(a.ckpt))
    clip = model.clip.to(dev).eval()
    tok = build_tokenizer(True)
    tf = init_transform_dict(224)["clip_test"]

    db = json.load(open(os.path.join(a.data_dir, "MSRVTT_data.json")))
    vids = sorted({v["video_id"] for v in db["videos"]}, key=lambda s: int(s[5:]))
    caps = {v: [] for v in vids}
    for s in db["sentences"]:
        caps[s["video_id"]].append(s["caption"])
    test = pd.read_csv(os.path.join(a.data_dir, "MSRVTT_JSFUSION_test.csv"))

    vfeat = torch.zeros(len(vids), 12, 512, dtype=torch.float16)
    for i in range(0, len(vids), a.bs_videos):
        chunk = vids[i:i + a.bs_videos]
        x = torch.stack([tf(torch.from_numpy(load_frames(a.cache_dir, v)).permute(0, 3, 1, 2).float() / 255)
                         for v in chunk]).to(dev)
        f = clip.get_image_features(x.reshape(-1, 3, 224, 224)).reshape(len(chunk), 12, -1)
        vfeat[i:i + len(chunk)] = f.half().cpu()
        if (i // a.bs_videos) % 20 == 0:
            print(f"videos {i}/{len(vids)}", flush=True)

    def encode(texts):
        out = []
        for i in range(0, len(texts), a.bs_text):
            t = tok(texts[i:i + a.bs_text], return_tensors="pt", padding=True, truncation=True)
            out.append(clip.get_text_features(**{k: v.to(dev) for k, v in t.items()}).half().cpu())
        return torch.cat(out)

    cap_list, cap_vid = [], []
    for vi, v in enumerate(vids):
        for c in caps[v]:
            cap_list.append(c)
            cap_vid.append(vi)
    tfeat = encode(cap_list)
    qfeat = encode(test["sentence"].tolist())
    torch.save({"video_ids": vids, "vid_feats": vfeat, "cap_texts": cap_list,
                "cap_vid_idx": torch.tensor(cap_vid), "cap_feats": tfeat,
                "test_video_ids": test["video_id"].tolist(), "test_texts": test["sentence"].tolist(),
                "test_feats": qfeat, "ckpt": a.ckpt, "logit_scale": float(clip.logit_scale.item())}, a.out)
    print("saved", a.out, vfeat.shape, tfeat.shape, qfeat.shape)


if __name__ == "__main__":
    main()
