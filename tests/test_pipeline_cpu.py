"""End-to-end smoke test on CPU with a tiny randomly initialised CLIP and synthetic data.

Checks that train (benchmark + full mini-run with evals/checkpointing/resume) and the
official-protocol evaluation run without errors on the current software stack, and that
checkpoints with legacy transformers-4.6 buffers load.
"""
import json
import os
import sys

import pandas as pd
import pytest
import torch


def tiny_clip():
    from transformers import CLIPConfig, CLIPModel
    cfg = CLIPConfig(
        projection_dim=512,
        text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                     "num_attention_heads": 2, "max_position_embeddings": 77},
        vision_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "image_size": 224, "patch_size": 32},
    )
    return CLIPModel(cfg)


class FakeTokenizer:
    """Mimics the HF CLIP tokenizer call signature; EOS gets the largest id (pooled position)."""

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        ids = [[49406] + [1 + (hash(w) % 49000) for w in t.split()][:20] + [49407] for t in texts]
        L = max(len(x) for x in ids)
        input_ids = torch.zeros(len(ids), L, dtype=torch.long)
        mask = torch.zeros(len(ids), L, dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


@pytest.fixture()
def tiny_world(synth_videos, tmp_path, monkeypatch):
    videos_dir, specs = synth_videos
    data_dir = tmp_path / "ann"
    data_dir.mkdir()
    vids = sorted(specs)
    sentences = []
    for v in vids:
        for j in range(3):
            sentences.append({"caption": f"a synthetic clip {v} number {j}", "video_id": v, "sen_id": len(sentences)})
    db = {"info": {}, "videos": [{"video_id": v, "category": i % 3, "split": "train"} for i, v in enumerate(vids)],
          "sentences": sentences}
    json.dump(db, open(data_dir / "MSRVTT_data.json", "w"))
    pd.DataFrame({"video_id": vids[:4]}).to_csv(data_dir / "MSRVTT_train.9k.csv", index=False)
    test = vids[4:]
    pd.DataFrame({"key": [f"ret{i}" for i in range(len(test))], "vid_key": test, "video_id": test,
                  "sentence": [f"a synthetic clip {v} number 0" for v in test]}).to_csv(
        data_dir / "MSRVTT_JSFUSION_test.csv", index=False)

    from tvr import frame_cache
    cache = tmp_path / "cache"
    frame_cache.build(videos_dir, str(cache), workers=2)

    # official _valid_epoch_step hard-codes .cuda(); make it a no-op on CPU-only test machines
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    import transformers
    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", staticmethod(lambda *a, **k: tiny_clip()))
    import tvr.compat as compat
    monkeypatch.setattr(compat, "build_tokenizer", lambda hf=True: FakeTokenizer())
    import tvr.train as train_mod
    import tvr.evaluate as eval_mod
    monkeypatch.setattr(train_mod, "build_tokenizer", lambda hf=True: FakeTokenizer())
    monkeypatch.setattr(eval_mod, "build_tokenizer", lambda hf=True: FakeTokenizer())
    common = ["--huggingface", "--dataset_name", "MSRVTT", "--msrvtt_train_file", "9k",
              "--data_dir", str(data_dir), "--cache_dir", str(cache), "--batch_size", "2",
              "--num_workers", "0", "--no_tensorboard", "--output_dir", str(tmp_path / "runs"),
              "--tb_log_dir", str(tmp_path / "tb")]
    return {"common": common, "tmp": tmp_path, "train": train_mod, "eval": eval_mod}


def _run(mod, argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"] + argv)
    mod.main()


def test_benchmark_mode(tiny_world, monkeypatch):
    w = tiny_world
    _run(w["train"], w["common"] + ["--exp_name", "bench", "--max_steps", "3", "--evals_per_epoch", "0"], monkeypatch)
    meta = json.load(open(w["tmp"] / "runs" / "bench" / "run_meta.json"))
    assert "benchmark" in meta and meta["benchmark"]["steps_measured"] >= 1
    assert meta["n_train_pairs"] == 12 and meta["steps_per_epoch"] == 6


def test_full_mini_run_eval_and_resume(tiny_world, monkeypatch):
    w = tiny_world
    _run(w["train"], w["common"] + ["--exp_name", "mini", "--num_epochs", "2", "--evals_per_epoch", "2"], monkeypatch)
    run = w["tmp"] / "runs" / "mini"
    for f in ["model_best.pth", "model_last.pth", "resume_state.pth", "eval_history.jsonl", "run_meta.json"]:
        assert (run / f).exists(), f
    hist = [json.loads(line) for line in open(run / "eval_history.jsonl")]
    assert sum(1 for h in hist if "R1" in h) == 4  # 2 epochs x 2 evals

    # evaluate best checkpoint with dumps and a temporal probe
    dump = w["tmp"] / "dump"
    for order in ["original", "shuffled"]:
        _run(w["eval"], w["common"] + ["--exp_name", "mini_eval", "--ckpt", str(run / "model_best.pth"),
                                       "--dump_dir", str(dump), "--frame_order", order], monkeypatch)
        res = json.load(open(dump / f"eval_{order}.json"))
        assert 0 <= res["t2v"]["R1"] <= 100
    d = torch.load(dump / "dump_original.pt", weights_only=False)
    assert d["sims"].shape == (2, 2) and len(d["texts"]) == 2

    # analysis tools run on the dumps
    import subprocess
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "src"))
    data_dir = [w["common"][i + 1] for i, x in enumerate(w["common"]) if x == "--data_dir"][0]
    cache_dir = [w["common"][i + 1] for i, x in enumerate(w["common"]) if x == "--cache_dir"][0]
    root = os.path.join(os.path.dirname(__file__), "..")
    subprocess.check_call([sys.executable, os.path.join(root, "analysis", "analyze_dump.py"), "--dump_dir", str(dump),
                           "--data_dir", data_dir, "--out_dir", str(w["tmp"] / "ana")], env=env)
    rep = json.load(open(w["tmp"] / "ana" / "analysis.json"))
    assert "frame_order_probes" in rep and "shuffled" in rep["frame_order_probes"]
    subprocess.check_call([sys.executable, os.path.join(root, "analysis", "failure_gallery.py"),
                           "--dump", str(dump / "dump_original.pt"), "--ckpt", str(run / "model_best.pth"),
                           "--cache_dir", cache_dir, "--out_dir", str(w["tmp"] / "gal"), "--n", "2",
                           "--select", "random"], env=env)

    # legacy transformers-4.6.1 buffers are tolerated, anything else is not
    from tvr.compat import load_xpool_state_dict
    from tvr.xpool_bridge import ensure_xpool_on_path
    ensure_xpool_on_path()
    from types import SimpleNamespace
    from model.clip_transformer import CLIPTransformer
    cfg = SimpleNamespace(huggingface=True, embed_dim=512, num_mha_heads=1, transformer_dropout=0.3,
                          input_res=224, num_frames=12)
    m = CLIPTransformer(cfg)
    sd = dict(torch.load(run / "model_best.pth", weights_only=False)["state_dict"])
    sd["clip.text_model.embeddings.position_ids"] = torch.arange(77)[None]
    sd["clip.vision_model.embeddings.position_ids"] = torch.arange(50)[None]
    info = load_xpool_state_dict(m, {"state_dict": sd}, verbose=False)
    assert len(info["dropped_legacy_keys"]) == 2
    sd["clip.some_unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        load_xpool_state_dict(m, {"state_dict": sd}, verbose=False)

    # resume from end of epoch 1 state file (written after epoch 2 here; just check it loads)
    meta1 = json.load(open(run / "run_meta.json"))
    assert meta1["n_sessions"] == 1 and all(h["session"] == 0 for h in hist if "R1" in h)
    _run(w["train"], w["common"] + ["--exp_name", "mini", "--num_epochs", "2", "--evals_per_epoch", "2",
                                    "--resume_state", str(run / "resume_state.pth")], monkeypatch)
    # session bookkeeping survives the resume (total time = all sessions, not only the last one)
    meta2 = json.load(open(run / "run_meta.json"))
    assert meta2["n_sessions"] == 2 and meta2["sessions"][0] == meta1["sessions"][0]
    assert meta2["sessions"][1]["resumed"] is True
    assert meta2["total_wall_hours"] >= meta1["total_wall_hours"]
    assert meta2["started_at"] == meta1["started_at"] and "best_checkpoint" in meta2


def test_runlog_upgrades_legacy_meta(tmp_path):
    """run_meta.json written before session tracking (seed 24 run) is kept as session 0."""
    from types import SimpleNamespace
    from tvr.runlog import RunLogger
    legacy = {"started_at": "2026-11-13 02:09:53", "finished_at": "2026-11-13 03:23:09", "total_wall_hours": 1.221,
              "train_seconds": 4356.1, "env": {"gpu_name": "NVIDIA A100-SXM4-80GB"}, "best_checkpoint": {"epoch": 5}}
    json.dump(legacy, open(tmp_path / "run_meta.json", "w"))
    rl = RunLogger(str(tmp_path), SimpleNamespace(exp_name="x", resume_state="r.pth"))
    meta = json.load(open(tmp_path / "run_meta.json"))
    assert meta["n_sessions"] == 2 and meta["sessions"][0]["wall_hours"] == 1.221
    assert meta["sessions"][0]["gpu_name"] == "NVIDIA A100-SXM4-80GB" and meta["best_checkpoint"] == {"epoch": 5}
    rl.log_eval({"R1": 1.0})
    assert json.loads(open(tmp_path / "eval_history.jsonl").readline())["session"] == 1


def test_proxy_extract_and_train(tiny_world, monkeypatch, tmp_path):
    """Feature extraction (zero-shot tiny CLIP) + proxy head training for every registered head."""
    w = tiny_world
    root = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, os.path.join(root, "proxy"))
    import extract_features as ef
    import proxy_train as pt
    monkeypatch.setattr(ef, "build_tokenizer", lambda hf=True: FakeTokenizer())
    data_dir = [w["common"][i + 1] for i, x in enumerate(w["common"]) if x == "--data_dir"][0]
    cache_dir = [w["common"][i + 1] for i, x in enumerate(w["common"]) if x == "--cache_dir"][0]
    feats = tmp_path / "feats.pt"
    _run(ef, ["--cache_dir", cache_dir, "--data_dir", data_dir, "--out", str(feats), "--bs_videos", "4"], monkeypatch)
    f = torch.load(feats, weights_only=False)
    assert f["vid_feats"].shape == (6, 12, 512) and f["test_feats"].shape[0] == 2
    for head in sorted(pt.HEADS):
        out = tmp_path / f"{head}.json"
        _run(pt, ["--feats", str(feats), "--head", head, "--epochs", "1", "--bs", "4", "--data_dir", data_dir,
                  "--out", str(out)], monkeypatch)
        r = json.load(open(out))
        assert 0 <= r["final"]["t2v"]["R1"] <= 100
