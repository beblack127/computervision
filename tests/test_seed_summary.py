"""seed_summary.py: mean / sample std over runs and markdown output."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import seed_summary as ss  # noqa: E402


def _run(root, name, seed, r1_best, r1_last):
    d = root / name
    d.mkdir()
    m = {"R1": 0, "R5": 70.0, "R10": 80.0, "MedR": 2.0, "MeanR": 14.0}
    for f, r1 in [("eval_original.json", r1_best), ("eval_last_original.json", r1_last)]:
        json.dump({"t2v": dict(m, R1=r1), "v2t": dict(m, R1=r1 - 1)}, open(d / f, "w"))
    json.dump({"config": {"seed": seed}, "env": {"gpu_name": "A100"}, "n_sessions": 1, "total_wall_hours": 6.0,
               "peak_gpu_mem_reserved_GB": 17.0, "best_checkpoint": {"global_step": 10}}, open(d / "run_meta.json", "w"))


def test_summary(tmp_path):
    for i, (b, l) in enumerate([(46.0, 46.0), (47.0, 46.5), (45.0, 44.0)]):
        _run(tmp_path, f"r{i}", i, b, l)
    S = ss.summarize(["r0", "r1", "r2"], str(tmp_path))
    assert abs(S["best_t2v"]["R1"]["mean"] - 46.0) < 1e-9 and abs(S["best_t2v"]["R1"]["std"] - 1.0) < 1e-9
    assert S["best_minus_last_t2v_R1"]["values"] == [0.0, 0.5, 1.0]
    assert "46.00 ± 1.00" in ss.to_md(S)
