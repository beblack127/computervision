"""Evaluate an X-Pool checkpoint on MSR-VTT 1k-A exactly like the official test.py,
and optionally dump everything needed for limitation analysis.

  # official protocol (t2v and v2t), official checkpoint:
  python -m tvr.evaluate --exp_name official_9k --ckpt /path/model_best.pth --cache_dir C --huggingface

  # analysis dumps + temporal-sensitivity probes:
  python -m tvr.evaluate ... --dump_dir D --frame_order shuffled

Outputs <dump_dir or run dir>/eval_<frame_order>.json with t2v and v2t metrics.
With --dump_dir: dump_<frame_order>.pt containing text/video embeddings, the full
1000x1000 similarity matrix, per-query ranks, captions and video ids.
"""
import json
import os
import time

import torch

from tvr.compat import build_tokenizer, load_xpool_state_dict, safe_torch_load
from tvr.config import build_parser, finalize
from tvr.data import get_data_loader
from tvr.runlog import environment_info
from tvr.xpool_bridge import REPO_ROOT, ensure_xpool_on_path

ensure_xpool_on_path()
from model.model_factory import ModelFactory  # noqa: E402  (official)
from modules.metrics import (generate_embeds_per_video_id, sim_matrix_inference,  # noqa: E402
                             t2v_metrics, v2t_metrics)


@torch.no_grad()
def run_eval(model, loader, tokenizer, device, pooling_type="transformer"):
    """Same computation as Trainer._valid_epoch_step (official), returning all intermediates."""
    model.eval()
    text_embed_arr, vid_embed_arr, all_vid_ids, all_texts = [], [], [], []
    for data in loader:
        all_texts.extend(list(data["text"]))
        data["text"] = tokenizer(data["text"], return_tensors="pt", padding=True, truncation=True)
        data["text"] = {k: v.to(device) for k, v in data["text"].items()}
        data["video"] = data["video"].to(device)
        text_embed, vid_embed, _ = model(data, return_all_frames=True)
        text_embed_arr.append(text_embed.cpu())
        vid_embed_arr.append(vid_embed.cpu())
        all_vid_ids.extend(list(data["video_id"]))
    text_embeds = torch.cat(text_embed_arr)
    vid_embeds = torch.cat(vid_embed_arr)
    vid_per_id = {}
    for idx, v_id in enumerate(all_vid_ids):
        if v_id not in vid_per_id:
            vid_per_id[v_id] = vid_embeds[idx]
    vid_embeds_u = torch.stack([vid_per_id[v] for v in vid_per_id])
    model.pool_frames.cpu()  # official does pooling for all pairs on CPU
    vid_embeds_pooled = model.pool_frames(text_embeds, vid_embeds_u)
    model.pool_frames.to(device)
    t_per_id, v_pooled_per_id = generate_embeds_per_video_id(text_embeds, vid_embeds_pooled,
                                                             all_vid_ids, pooling_type)
    sims = sim_matrix_inference(t_per_id, v_pooled_per_id, pooling_type)
    res_t2v = t2v_metrics(sims.clone())
    res_v2t = v2t_metrics(sims.clone())
    return {"t2v": res_t2v, "v2t": res_v2t, "sims": sims, "text_embeds": text_embeds,
            "vid_embeds": vid_embeds_u, "video_ids": list(vid_per_id.keys()),
            "query_video_ids": all_vid_ids, "texts": all_texts}


def main():
    p = build_parser("Evaluate X-Pool (project wrapper)")
    p.add_argument("--ckpt", type=str, required=True, help="model_best.pth / model_last.pth / official file")
    p.add_argument("--frame_order", default="original", choices=["original", "reversed", "shuffled", "middle"])
    p.add_argument("--dump_dir", type=str, default=None)
    args = p.parse_args()
    config = finalize(args)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(config.huggingface)
    loader = get_data_loader(config, "test", frame_order=config.frame_order)
    model = ModelFactory.get_model(config)
    load_info = load_xpool_state_dict(model, safe_torch_load(config.ckpt))
    model.to(device)

    t0 = time.time()
    out = run_eval(model, loader, tokenizer, device, config.pooling_type or "transformer")
    secs = round(time.time() - t0, 1)
    summary = {"ckpt": config.ckpt, "frame_order": config.frame_order, "eval_seconds": secs,
               "t2v": out["t2v"], "v2t": out["v2t"], "load_info": load_info,
               "env": environment_info(REPO_ROOT)}
    out_dir = config.dump_dir or config.model_path
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"eval_{config.frame_order}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({"t2v": out["t2v"], "v2t": out["v2t"], "seconds": secs}, indent=2, default=float))

    if config.dump_dir:
        sims = out["sims"]
        if sims.dim() == 3 and sims.shape[1] == 1:  # 1k-A: one caption per video
            sims = sims[:, 0, :]
        torch.save({"sims": sims, "text_embeds": out["text_embeds"], "vid_embeds": out["vid_embeds"],
                    "video_ids": out["video_ids"], "query_video_ids": out["query_video_ids"],
                    "texts": out["texts"], "frame_order": config.frame_order, "ckpt": config.ckpt},
                   os.path.join(config.dump_dir, f"dump_{config.frame_order}.pt"))
        print(f"[evaluate] dump written to {config.dump_dir}")


if __name__ == "__main__":
    main()
