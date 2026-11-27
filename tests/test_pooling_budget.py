"""pooling_budget.py: U-Pool / alpha sweep equal the official math, two-stage limits, label-free budgets."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from tvr.xpool_bridge import ensure_xpool_on_path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "analysis"))


def make_run(root, name="run_a", Q=40, seed=0):
    ensure_xpool_on_path()
    from modules.transformer import Transformer
    import cost_tradeoff as ct
    torch.manual_seed(seed)
    pool = Transformer(SimpleNamespace(embed_dim=512, num_mha_heads=1, transformer_dropout=0.3)).eval()
    with torch.no_grad():
        for p_ in pool.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    text = torch.randn(Q, 512)
    frames = text[:, None, :] + 4.0 * torch.randn(Q, 12, 512)  # hard enough that R@1 < 100
    sims = ct.xpool_sims(pool, text, frames)
    vids = [f"video{i}" for i in range(Q)]
    d = root / name
    (d / "dump_model_best").mkdir(parents=True)
    torch.save({"sims": sims, "text_embeds": text, "vid_embeds": frames, "video_ids": vids,
                "query_video_ids": vids, "texts": [f"q{i}" for i in range(Q)]}, d / "dump_model_best" / "dump_original.pt")
    torch.save({"state_dict": {f"pool_frames.{k}": v for k, v in pool.state_dict().items()}}, d / "model_best.pth")
    return d, pool, text, frames, sims


def test_alpha_and_upool_match_official(tmp_path):
    import pooling_budget as pb
    _, pool, text, frames, sims = make_run(tmp_path)
    assert torch.allclose(pb.xpool_sims_alpha(pool, text, frames, 1.0, chunk=7), sims, atol=1e-5)
    s_u = pb.upool_sims(pool, text, frames)
    assert torch.allclose(pb.xpool_sims_alpha(pool, text, frames, 0.0), s_u, atol=1e-5)
    # U-Pool is query-agnostic: the pooled video vector does not depend on the text
    e1 = pb.upool_embed(pool, frames)
    out = pool(text[:3], frames[:5])  # official forward with real attention: [V, Q, D]
    assert out.shape == (5, 3, 512) and e1.shape == (frames.shape[0], 512)


def test_two_stage_var_limits(tmp_path):
    import pooling_budget as pb
    import cost_tradeoff as ct
    _, pool, text, frames, s_x = make_run(tmp_path)
    s_m = ct.mean_pool_sims(text, frames)
    V = s_x.shape[1]
    assert (pb.two_stage_var(s_m, s_x, V) == ct.ranks_of(s_x)).all()   # full shortlist = X-Pool
    assert (pb.two_stage_var(s_m, s_x, 0) == ct.ranks_of(s_m)).all()   # no re-ranking = first stage
    for P in [1, 5, 13]:
        assert (pb.two_stage_var(s_m, s_x, P) == ct.two_stage(s_m, s_x, P)).all()
    Pq = np.array([0, 3] * (V // 2))
    r = pb.two_stage_var(s_m, s_x, Pq)
    assert (r[Pq == 0] == ct.ranks_of(s_m)[Pq == 0]).all()


def test_adaptive_budget_is_label_free_and_exact():
    import pooling_budget as pb
    gap = np.random.default_rng(0).random(1000)
    for B in [2, 5, 10, 20]:
        P = pb.adaptive_P(gap, B, 0, 50)
        assert abs(P.mean() - B) < 0.06
        assert P[np.argmin(gap)] == 50 and P[np.argmax(gap)] == 0   # least confident query gets the budget


def test_script_end_to_end(tmp_path):
    a, *_ = make_run(tmp_path, "run_a", seed=0)
    b, *_ = make_run(tmp_path, "run_b", seed=1)
    for d in (a, b):  # querybank of "training" captions
        torch.save({"text_embeds": torch.randn(60, 512)}, d / "querybank_9k_5000.pt")
    out = tmp_path / "out"
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.check_call([sys.executable, os.path.join(ROOT, "analysis", "pooling_budget.py"), "--runs", str(a), str(b),
                           "--out_dir", str(out), "--fixed_P", "1", "5", "40", "--budgets", "2", "5",
                           "--adaptive", "0:20", "--gallery_sizes", "40", "80", "--timing_P", "5"], env=env)
    R = json.load(open(out / "run_a.json"))
    assert R["sanity_alpha1_vs_dump_max_abs_diff"] < 1e-4 and R["sanity_alpha0_vs_upool_max_abs_diff"] < 1e-4
    assert R["two_stage"]["E1_mean_fixed"][-1]["R1"] == R["scoring"]["xpool"]["R1"]
    S = json.load(open(out / "summary.json"))
    assert S["n"] == 2 and len(S["timing_first_run"]) == 2
    acc = S["accuracy_no_training"]
    assert "xpool + QB-Norm" in acc and "fusion xpool+1.0*upool" in acc
    assert abs(acc["xpool"]["R1"]["mean"] - S["scoring"]["xpool"]["R1"]["mean"]) < 1e-9
    for f in ["pooling_budget_report.md", "r1_vs_budget.png", "r1_vs_alpha.png"]:
        assert (out / f).exists(), f


def test_qb_norm_dis_properties():
    import pooling_budget as pb
    import cost_tradeoff as ct
    g = torch.Generator().manual_seed(0)
    s_q = torch.rand(30, 30, generator=g)
    flat_bank = torch.full((50, 30), 0.3)                    # no item is special -> rankings unchanged
    out, _ = pb.qb_norm_dis(s_q, flat_bank)
    assert (ct.ranks_of(out) == ct.ranks_of(s_q)).all()
    hub_bank = torch.rand(50, 30, generator=g) * 0.1
    hub_bank[:, 7] = 0.9                                     # video 7 is close to every bank query
    s = s_q.clone(); s[:, 7] = s.max(1).values + 0.01        # ... and wins every test query
    out, frac = pb.qb_norm_dis(s, hub_bank)
    assert frac == 1.0 and (out.argmax(1) != 7).all()        # the hub is demoted


def test_sample_captions(tmp_path):
    import embed_querybank as eq
    (tmp_path / "MSRVTT_train.9k.csv").write_text("video_id\nvideo1\nvideo2\n")
    (tmp_path / "MSRVTT_JSFUSION_test.csv").write_text("key,vid_key,video_id,sentence\nk,v,video9,s\n")
    sents = [{"video_id": v, "caption": f"{v} cap {i}", "sen_id": i} for i in range(5) for v in ["video1", "video2", "video9"]]
    json.dump({"videos": [], "sentences": sents}, open(tmp_path / "MSRVTT_data.json", "w"))
    texts, vids = eq.sample_captions(str(tmp_path), n=6, seed=0)
    assert len(texts) == 6 and "video9" not in vids
    assert eq.sample_captions(str(tmp_path), n=6, seed=0)[0] == texts   # reproducible
