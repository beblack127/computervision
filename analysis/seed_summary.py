"""Multi-seed reproduction table (mean ± sample std) from experiments/results/<run>/ files.

  python analysis/seed_summary.py --runs xpool_9k_s24 xpool_9k_s0 xpool_9k_s1 \
      --out experiments/results/reproduction_3seed

Reads eval_original.json (best-on-test checkpoint), eval_last_original.json (last step),
run_meta.json (time, memory, best step) and go_no_go.json (official checkpoint re-evaluated
with our pipeline; equals the paper's t2v numbers). Writes <out>.json and <out>.md (tables only).
"""
import argparse
import json
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")
KEYS = ["R1", "R5", "R10", "MedR", "MeanR"]


def load(p):
    with open(p) as f:
        return json.load(f)


def agg(vals):
    return {"mean": st.mean(vals), "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "values": vals}


def summarize(runs, root=ROOT):
    per_run = {}
    for r in runs:
        d = os.path.join(root, r)
        best, last, meta = (load(os.path.join(d, f)) for f in
                            ["eval_original.json", "eval_last_original.json", "run_meta.json"])
        sess = meta.get("sessions", [])
        recon = os.path.join(d, "sessions_reconstructed.json")
        per_run[r] = {
            "seed": meta["config"]["seed"],
            "best": {"t2v": best["t2v"], "v2t": best["v2t"]},
            "last": {"t2v": last["t2v"], "v2t": last["v2t"]},
            "best_checkpoint": meta.get("best_checkpoint"),
            "gpus": meta.get("gpus_used") or [meta["env"]["gpu_name"]],
            "n_sessions": meta.get("n_sessions", len(sess)),
            "total_wall_hours": meta.get("total_wall_hours"),
            "wall_hours_note": "",
            "peak_gpu_mem_reserved_GB": meta.get("peak_gpu_mem_reserved_GB_all_sessions",
                                                 meta.get("peak_gpu_mem_reserved_GB")),
            "sec_per_step_median": (meta.get("sec_per_step") or {}).get("sec_per_step_median"),
        }
        if os.path.exists(recon):  # run_meta of this run covers only its last session
            rc = load(recon)
            per_run[r].update({"n_sessions": rc["n_sessions"],
                               "total_wall_hours": rc.get("total_wall_hours_lower_bound"),
                               "wall_hours_note": "lower bound, reconstructed (sessions_reconstructed.json)"})
    out = {"runs": per_run, "n_seeds": len(runs)}
    for sel in ["best", "last"]:
        for task in ["t2v", "v2t"]:
            out[f"{sel}_{task}"] = {k: agg([per_run[r][sel][task][k] for r in runs]) for k in KEYS}
    out["best_minus_last_t2v_R1"] = agg([round(per_run[r]["best"]["t2v"]["R1"] - per_run[r]["last"]["t2v"]["R1"], 2)
                                         for r in runs])
    g = os.path.join(root, "go_no_go.json")
    if os.path.exists(g):
        gg = load(g)
        out["official_ckpt"] = {"t2v": gg["official_ckpt_t2v"], "v2t": gg["official_ckpt_v2t"],
                                "note": "official X-Pool 9k checkpoint re-evaluated with our pipeline (= paper t2v)"}
    return out


def to_md(S):
    runs = list(S["runs"])
    L = ["# X-Pool reproduction over seeds (auto-generated)", "",
         "best = checkpoint selected on the test set (official protocol); last = final step. std = sample std.", ""]
    for sel in ["best", "last"]:
        for task in ["t2v", "v2t"]:
            L += [f"## {task} · {sel}", "", "| run | seed | " + " | ".join(KEYS) + " |",
                  "|---|---|" + "---|" * len(KEYS)]
            for r in runs:
                m = S["runs"][r][sel][task]
                L.append(f"| {r} | {S['runs'][r]['seed']} | " + " | ".join(f"{m[k]:.1f}" for k in KEYS) + " |")
            a = S[f"{sel}_{task}"]
            L.append(f"| **mean ± std** | | " + " | ".join(f"{a[k]['mean']:.2f} ± {a[k]['std']:.2f}" for k in KEYS) + " |")
            if "official_ckpt" in S:
                o = S["official_ckpt"][task]
                L.append("| official ckpt (ours eval) | 24 | " + " | ".join(f"{o[k]:.1f}" for k in KEYS) + " |")
            L.append("")
    b = S["best_minus_last_t2v_R1"]
    L += [f"best − last (t2v R@1): {b['values']} → mean {b['mean']:.2f}", "",
          "## Cost per run", "", "| run | GPU | sessions | wall h | peak reserved GB | s/step (median) | best step |",
          "|---|---|---|---|---|---|---|"]
    for r, v in S["runs"].items():
        bc = v["best_checkpoint"] or {}
        sps = f"{v['sec_per_step_median']:.3f}" if v["sec_per_step_median"] else "–"
        L.append(f"| {r} | {', '.join(v['gpus'])} | {v['n_sessions']} | {v['total_wall_hours']}"
                 f"{' (≥) *' if v['wall_hours_note'] else ''} | {v['peak_gpu_mem_reserved_GB']} | {sps} | {bc.get('global_step', '–')} |")
    L += ["", "\\* reconstructed lower bound: run_meta of that run covered only its last session (sessions_reconstructed.json)"]
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True, help="output path without extension")
    p.add_argument("--root", default=ROOT)
    a = p.parse_args()
    S = summarize(a.runs, a.root)
    json.dump(S, open(a.out + ".json", "w"), indent=2)
    open(a.out + ".md", "w").write(to_md(S))
    print(to_md(S))


if __name__ == "__main__":
    main()
