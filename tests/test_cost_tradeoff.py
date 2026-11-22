"""cost_tradeoff.py on a synthetic dump: sanity check, two-stage limits, outputs."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from tvr.xpool_bridge import ensure_xpool_on_path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis"))


def make_world(tmp_path, Q=30):
    ensure_xpool_on_path()
    from modules.transformer import Transformer
    torch.manual_seed(0)
    pool = Transformer(SimpleNamespace(embed_dim=512, num_mha_heads=1, transformer_dropout=0.3)).eval()
    with torch.no_grad():
        for p_ in pool.parameters():
            p_.add_(0.02 * torch.randn_like(p_))
    text = torch.randn(Q, 512)
    frames = text[:, None, :] + 0.8 * torch.randn(Q, 12, 512)  # GT video correlated with its query
    import cost_tradeoff as ct
    sims = ct.xpool_sims(pool, text, frames)
    vids = [f"video{i}" for i in range(Q)]
    torch.save({"sims": sims, "text_embeds": text, "vid_embeds": frames, "video_ids": vids,
                "query_video_ids": vids, "texts": [f"a person then runs {i}" if i % 3 == 0 else f"a dog {i}"
                                                   for i in range(Q)]}, tmp_path / "dump.pt")
    torch.save({"state_dict": {f"pool_frames.{k}": v for k, v in pool.state_dict().items()}}, tmp_path / "ckpt.pth")
    data = tmp_path / "ann"
    data.mkdir()
    json.dump({"videos": [{"video_id": v, "category": i % 4} for i, v in enumerate(vids)], "sentences": []},
              open(data / "MSRVTT_data.json", "w"))
    return ct, pool, text, frames, sims, data


def test_two_stage_limits(tmp_path):
    ct, pool, text, frames, s_x, _ = make_world(tmp_path)
    s_m = ct.mean_pool_sims(text, frames)
    full = ct.two_stage(s_m, s_x, s_x.shape[1])
    assert (full == ct.ranks_of(s_x)).all()        # P = all videos -> exactly X-Pool
    one = ct.two_stage(s_m, s_x, 1)
    assert ((one == 0) == (ct.ranks_of(s_m) == 0)).all()  # P = 1 -> top-1 decided by mean pooling


def test_script_end_to_end(tmp_path):
    make_world(tmp_path)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    out = tmp_path / "out"
    subprocess.check_call([sys.executable, os.path.join(ROOT, "analysis", "cost_tradeoff.py"),
                           "--dump", str(tmp_path / "dump.pt"), "--ckpt", str(tmp_path / "ckpt.pth"),
                           "--data_dir", str(tmp_path / "ann"), "--out_dir", str(out),
                           "--P", "1", "5", "30", "--P_cost", "5", "--gallery_sizes", "30", "60",
                           "--cost_queries", "10"], env=env)
    R = json.load(open(out / "cost_tradeoff.json"))
    assert R["sanity_max_abs_diff_xpool_sims"] < 1e-4
    assert R["two_stage"][-1]["R1"] == R["accuracy"]["xpool"]["R1"]
    assert len(R["cost_scaling"]) == 2 and R["cost_scaling"][1]["xpool_pairs"] == 600
    assert sum(v["n"] for v in R["breakdown"]["frame_diversity_quartile"].values()) == 30
    for f in ["cost_report.md", "two_stage_r1_vs_P.png", "cost_scaling.png", "gain_by_diversity.png"]:
        assert (out / f).exists(), f
