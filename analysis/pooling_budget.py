"""Query-dependent pooling on a budget (Intermediate: limitation L2/L3 + initial idea).

Inference only, no training. For each trained run (dump + model_best.pth):

1. Attention-sharpness sweep (own analysis). X-Pool's attention logits are multiplied by alpha:
   alpha = 1 is the official X-Pool; alpha = 0 gives uniform attention, which makes the whole trained
   pooling head query-agnostic -> "U-Pool": the X-Pool head with uniform attention, i.e. ONE vector per
   video that can be pre-computed and indexed like mean pooling. Mean -> U-Pool -> X-Pool splits X-Pool's
   gain into (learned projection layers, query-agnostic) and (query-conditioned attention).
2. Two-stage retrieval: X-Pool re-ranks only a shortlist from a cheap first stage.
   Families (accuracy vs. X-Pool pair evaluations per query):
     E1  mean first stage, fixed shortlist P            [existing: X-Pool paper App. C; Bain et al. 2022]
     E2  mean first stage, query-adaptive shortlist     [existing idea, adapted: ranked-list truncation /
         (P_hi for the least confident queries,           dynamic re-ranking cutoff, e.g. Culpepper et al.
          P_lo for the rest; confidence = top1-top2 gap)  2016, Lien et al. 2019, Meng et al. 2024]
     O1  U-Pool first stage, fixed P                    [own]
     O2  U-Pool first stage, query-adaptive shortlist   [own first stage + adapted depth rule]
   Adaptive thresholds are chosen from first-stage scores only (no labels) to hit a target average budget,
   so no test-set tuning is involved.
3. Hubness of mean / U-Pool / X-Pool scores, and GPU timing of the ranking stage.

  python analysis/pooling_budget.py --runs RUNS/xpool_9k_s24 RUNS/xpool_9k_s0 RUNS/xpool_9k_s1 \
      --out_dir RUNS/pooling_budget
(each run dir must contain dump_model_best/dump_original.pt and model_best.pth)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from cost_tradeoff import mean_pool_sims, metrics, ranks_of, xpool_kv, xpool_pair_sims, xpool_sims  # noqa: E402
from failure_gallery import load_pooling  # noqa: E402


# ---------------------------------------------------------------- scoring rules
@torch.no_grad()
def _head(pool, att):
    """Everything after the attention-weighted sum (official Transformer.forward): [.., D] -> [.., D]."""
    o = pool.layer_norm2(pool.cross_attn.out_proj(att))
    return pool.layer_norm3(o + pool.dropout(pool.linear_proj(o)))


@torch.no_grad()
def xpool_sims_alpha(pool, text, frames, alpha=1.0, chunk=64):
    """X-Pool scores [Q, V] with attention logits scaled by alpha (alpha=1: official X-Pool,
    alpha=0: uniform attention). Same math as third_party/xpool/modules/transformer.py."""
    ca = pool.cross_attn
    H, hd = ca.num_heads, ca.head_dim
    Q, D = text.shape
    k, v = xpool_kv(pool, frames)                                 # [V, H, F, hd]
    q = ca.q_proj(pool.layer_norm1(text)).view(Q, H, hd)          # [Q, H, hd]
    tn = F.normalize(text, dim=-1)
    out = []
    for i in range(0, frames.shape[0], chunk):
        kk, vv = k[i:i + chunk], v[i:i + chunk]
        logits = torch.einsum("qhd,nhfd->nqhf", q, kk) * (alpha / math.sqrt(hd))
        w = F.softmax(logits, dim=-1)
        att = torch.einsum("nqhf,nhfd->nqhd", w, vv).reshape(kk.shape[0], Q, D)
        out.append(torch.einsum("qd,nqd->qn", tn, F.normalize(_head(pool, att), dim=-1)))
    return torch.cat(out, 1)


@torch.no_grad()
def upool_embed(pool, frames):
    """U-Pool: trained X-Pool head with uniform attention -> one query-agnostic vector per video [V, D]."""
    k, v = xpool_kv(pool, frames)                                 # [V, H, F, hd]
    att = v.mean(2).reshape(frames.shape[0], -1)                  # uniform weights over frames
    return _head(pool, att)


def upool_sims(pool, text, frames):
    return F.normalize(text, dim=-1) @ F.normalize(upool_embed(pool, frames), dim=-1).t()


# ---------------------------------------------------------------- two-stage retrieval
def two_stage_var(s1, s2, P):
    """Per-query shortlist size P[q] (0..V) from first-stage scores s1, re-ranked by s2 inside the
    shortlist; items outside keep their first-stage order behind it. P[q]=0: no re-ranking.
    Returns 0-based ground-truth ranks (GT of query i is video i)."""
    Q, V = s1.shape
    P = torch.as_tensor(np.broadcast_to(np.asarray(P), (Q,)).copy()).clamp(0, V)
    order = torch.argsort(s1, dim=1, descending=True)
    pos = torch.empty_like(order)
    pos.scatter_(1, order, torch.arange(V).expand(Q, V))          # pos[q, v] = first-stage position of v
    gt = torch.arange(Q)
    pos_gt = pos[gt, gt]
    in_short = pos < P[:, None]
    better = (s2 > s2[gt, gt][:, None]) & in_short
    rank_in = better.sum(1)
    return torch.where(pos_gt < P, rank_in, pos_gt).numpy()


def score_gap(s1):
    top2 = torch.topk(s1, 2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).numpy()


def adaptive_P(gap, budget, p_lo, p_hi):
    """Label-free policy: the least confident queries (smallest top1-top2 gap) get p_hi, the rest p_lo,
    with the hard fraction set so that the average shortlist size equals `budget`."""
    frac = float(np.clip((budget - p_lo) / (p_hi - p_lo), 0.0, 1.0))
    n_hard = int(round(frac * len(gap)))
    P = np.full(len(gap), p_lo, dtype=np.int64)
    if n_hard > 0:
        P[np.argsort(gap, kind="stable")[:n_hard]] = p_hi
    return P


# ---------------------------------------------------------------- hubness
def hubness(sims, k=10):
    Q, V = sims.shape
    topk = torch.topk(sims, k, dim=1).indices
    nk = torch.bincount(topk.flatten(), minlength=V).float().numpy()
    n1 = torch.bincount(topk[:, 0], minlength=V).numpy()
    m, s = nk.mean(), nk.std()
    return {"N_k_skewness": float(((nk - m) ** 3).mean() / (s ** 3 + 1e-12)), "max_N1": int(n1.max()),
            "videos_never_top1": int((n1 == 0).sum()), "k": k}


# ---------------------------------------------------------------- timing
@torch.no_grad()
def _time(fn, device, repeat=3):
    fn()  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


@torch.no_grad()
def timing(pool, text, frames, device, gallery_sizes, fixed_P, budget, p_lo, p_hi):
    pool = pool.to(device).eval()
    q = text.float().to(device)
    tq = F.normalize(q, dim=-1)
    res = []
    for N in gallery_sizes:
        reps = math.ceil(N / frames.shape[0])
        fr = frames.float().repeat(reps, 1, 1)[:N].to(device)
        t_index = _time(lambda: F.normalize(upool_embed(pool, fr), dim=-1), device, repeat=1)  # offline
        idx = F.normalize(upool_embed(pool, fr), dim=-1)
        row = {"gallery": N, "queries": int(q.shape[0]), "upool_index_build_ms_offline": t_index,
               "first_stage_ms": _time(lambda: torch.topk(tq @ idx.t(), max(fixed_P), dim=1), device),
               "xpool_all_pairs_ms": _time(lambda: xpool_sims(pool, q, fr, 128), device)}
        for P in fixed_P:
            row[f"two_stage_P{P}_ms"] = _time(
                lambda: xpool_pair_sims(pool, q, fr, torch.topk(tq @ idx.t(), P, dim=1).indices), device)

        def adaptive():
            s1 = tq @ idx.t()
            Pq = torch.as_tensor(adaptive_P(score_gap(s1.cpu()), budget, p_lo, p_hi), device=device)
            for p in (p_lo, p_hi):
                sel = (Pq == p).nonzero().flatten()
                if p > 0 and len(sel):
                    xpool_pair_sims(pool, q[sel], fr, torch.topk(s1[sel], p, dim=1).indices)
        row[f"adaptive_B{budget}_P{p_lo}-{p_hi}_ms"] = _time(adaptive, device)
        res.append(row)
        print("[timing]", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        del fr
    return res


# ---------------------------------------------------------------- one run
def analyze_run(run_dir, a, device):
    d = torch.load(os.path.join(run_dir, "dump_model_best", "dump_original.pt"), weights_only=False)
    assert list(d["query_video_ids"]) == list(d["video_ids"]), "expects 1k-A layout"
    text, frames, s_x = d["text_embeds"].float(), d["vid_embeds"].float(), d["sims"].float()
    pool = load_pooling(os.path.join(run_dir, "model_best.pth"))
    R = {"run": os.path.basename(os.path.normpath(run_dir))}

    s_m = mean_pool_sims(text, frames)
    s_u = upool_sims(pool, text, frames)
    R["sanity_alpha1_vs_dump_max_abs_diff"] = float((xpool_sims_alpha(pool, text[:64], frames) - s_x[:64]).abs().max())
    R["sanity_alpha0_vs_upool_max_abs_diff"] = float((xpool_sims_alpha(pool, text[:64], frames, 0.0) - s_u[:64]).abs().max())

    rx, rm, ru = ranks_of(s_x), ranks_of(s_m), ranks_of(s_u)
    R["scoring"] = {"mean": metrics(rm), "upool": metrics(ru), "xpool": metrics(rx)}
    R["alpha_sweep"] = []
    for al in a.alphas:
        s = s_x if al == 1.0 else (s_u if al == 0.0 else xpool_sims_alpha(pool, text, frames, al))
        R["alpha_sweep"].append(dict(metrics(ranks_of(s)), alpha=al))
    c = {"mean": rm == 0, "upool": ru == 0, "xpool": rx == 0}
    R["top1_transitions"] = {
        "correct_mean": int(c["mean"].sum()), "correct_upool": int(c["upool"].sum()), "correct_xpool": int(c["xpool"].sum()),
        "fixed_by_projection(mean wrong, upool right)": int((~c["mean"] & c["upool"]).sum()),
        "broken_by_projection(mean right, upool wrong)": int((c["mean"] & ~c["upool"]).sum()),
        "fixed_by_attention(upool wrong, xpool right)": int((~c["upool"] & c["xpool"]).sum()),
        "broken_by_attention(upool right, xpool wrong)": int((c["upool"] & ~c["xpool"]).sum()),
        "oracle_any_of_three": int((c["mean"] | c["upool"] | c["xpool"]).sum()),
    }
    R["hubness"] = {"mean": hubness(s_m), "upool": hubness(s_u), "xpool": hubness(s_x)}

    fam = {"E1_mean_fixed": [], "O1_upool_fixed": []}
    for name, s1 in [("E1_mean_fixed", s_m), ("O1_upool_fixed", s_u)]:
        for P in a.fixed_P:
            fam[name].append(dict(metrics(two_stage_var(s1, s_x, P)), budget=float(min(P, s1.shape[1])), P=P))
    for lo, hi in a.adaptive:
        for tag, s1 in [("E2_mean_adaptive", s_m), ("O2_upool_adaptive", s_u)]:
            key = f"{tag}_P{lo}-{hi}"
            fam[key] = []
            g = score_gap(s1)
            for B in a.budgets:
                if not lo <= B <= hi:
                    continue
                Pq = adaptive_P(g, B, lo, hi)
                fam[key].append(dict(metrics(two_stage_var(s1, s_x, Pq)), budget=float(Pq.mean()), target=B))
    R["two_stage"] = fam
    if not a.skip_timing:
        lo, hi = a.adaptive[0]
        R["timing"] = timing(pool, text, frames, device, a.gallery_sizes, a.timing_P, a.timing_budget, lo, hi)
        if device.type == "cuda":
            p = torch.cuda.get_device_properties(device)
            R["gpu"] = {"name": p.name, "total_memory_GB": round(p.total_memory / 1024 ** 3, 1)}
    return R


# ---------------------------------------------------------------- summary over runs
def _ms(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "values": v.tolist()}


def summarize(runs):
    S = {"runs": [r["run"] for r in runs], "n": len(runs)}
    S["scoring"] = {k: {m: _ms([r["scoring"][k][m] for r in runs]) for m in ["R1", "R5", "R10", "MnR"]}
                    for k in ["mean", "upool", "xpool"]}
    S["alpha_sweep"] = [{"alpha": row["alpha"], **{m: _ms([r["alpha_sweep"][i][m] for r in runs]) for m in ["R1", "R5", "R10"]}}
                        for i, row in enumerate(runs[0]["alpha_sweep"])]
    S["top1_transitions"] = {k: _ms([r["top1_transitions"][k] for r in runs]) for k in runs[0]["top1_transitions"]}
    S["hubness"] = {k: {m: _ms([r["hubness"][k][m] for r in runs]) for m in ["N_k_skewness", "max_N1", "videos_never_top1"]}
                    for k in ["mean", "upool", "xpool"]}
    S["two_stage"] = {}
    for fam, rows in runs[0]["two_stage"].items():
        S["two_stage"][fam] = [{"budget": _ms([r["two_stage"][fam][i]["budget"] for r in runs]),
                                **{m: _ms([r["two_stage"][fam][i][m] for r in runs]) for m in ["R1", "R5", "R10", "MnR"]},
                                **({"P": rows[i]["P"]} if "P" in rows[i] else {"target": rows[i]["target"]})}
                               for i in range(len(rows))]
    if "timing" in runs[0]:  # timing is measured on the first run only
        S["timing_first_run"] = runs[0]["timing"]
        S["gpu"] = runs[0].get("gpu")
    return S


def f2(x):
    return f"{x['mean']:.1f} ± {x['std']:.1f}"


def to_md(S):
    L = [f"# Query-dependent pooling on a budget (auto-generated, {S['n']} runs: {', '.join(S['runs'])})", "",
         "mean ± sample std over runs. U-Pool = trained X-Pool head with uniform attention (query-agnostic, "
         "pre-computable). budget = X-Pool pair evaluations per query.", "",
         "## 1. Where does X-Pool's gain come from? (same trained model, three scoring rules)", "",
         "| scoring | query-dependent? | index per video | R@1 | R@5 | R@10 | MnR |", "|---|---|---|---|---|---|---|"]
    for k, qd, idx in [("mean", "no", "1 vector"), ("upool", "no", "1 vector"), ("xpool", "yes", "12 frame vectors")]:
        s = S["scoring"][k]
        L.append(f"| {k} | {qd} | {idx} | {f2(s['R1'])} | {f2(s['R5'])} | {f2(s['R10'])} | {f2(s['MnR'])} |")
    L += ["", "| top-1 transitions (count of 1000 queries) | mean ± std |", "|---|---|"]
    L += [f"| {k} | {f2(v)} |" for k, v in S["top1_transitions"].items()]
    L += ["", "## 2. Attention sharpness (logits × alpha; 0 = uniform = U-Pool, 1 = X-Pool)", "",
          "| alpha | R@1 | R@5 | R@10 |", "|---|---|---|---|"]
    L += [f"| {r['alpha']} | {f2(r['R1'])} | {f2(r['R5'])} | {f2(r['R10'])} |" for r in S["alpha_sweep"]]
    L += ["", "## 3. Hubness (k=10)", "", "| scoring | N_k skewness | max N1 | videos never top-1 |", "|---|---|---|---|"]
    for k, h in S["hubness"].items():
        L.append(f"| {k} | {h['N_k_skewness']['mean']:.2f} ± {h['N_k_skewness']['std']:.2f} | {f2(h['max_N1'])} | {f2(h['videos_never_top1'])} |")
    L += ["", "## 4. Two-stage retrieval: accuracy vs X-Pool budget", "",
          "E1 = mean→X-Pool fixed P (X-Pool App. C; Bain et al. 2022) · E2 = mean→X-Pool adaptive depth "
          "(ranked-list truncation, adapted) · O1/O2 = U-Pool first stage (own).", ""]
    for fam, rows in S["two_stage"].items():
        L += [f"### {fam}", "", "| P / target | avg pairs/query | R@1 | R@5 | R@10 | MnR |", "|---|---|---|---|---|---|"]
        L += [f"| {r.get('P', r.get('target'))} | {r['budget']['mean']:.1f} | {f2(r['R1'])} | {f2(r['R5'])} | {f2(r['R10'])} | {f2(r['MnR'])} |"
              for r in rows]
        L.append("")
    if "timing_first_run" in S:
        g = S.get("gpu") or {}
        L += [f"## 5. Ranking-stage time (first run, {g.get('name', 'cpu')}, 1000 queries, ms)", ""]
        keys = [k for k in S["timing_first_run"][0] if k not in ("gallery", "queries")]
        L += ["| gallery | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        L += [f"| {r['gallery']} | " + " | ".join(f"{r[k]:.2f}" for k in keys) + " |" for r in S["timing_first_run"]]
        L.append("")
    return "\n".join(L) + "\n"


def plots(S, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print("plotting skipped:", e)
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    style = {"E1": ("C0", "o", "-"), "E2": ("C0", "s", "--"), "O1": ("C3", "o", "-"), "O2": ("C3", "s", "--")}
    for fam, rows in S["two_stage"].items():
        c, m, ls = style[fam[:2]]
        rows = [r for r in rows if r["budget"]["mean"] >= 1]
        ax.errorbar([r["budget"]["mean"] for r in rows], [r["R1"]["mean"] for r in rows],
                    yerr=[r["R1"]["std"] for r in rows], color=c, marker=m, ls=ls, ms=4, capsize=2, label=fam)
    ax.axhline(S["scoring"]["xpool"]["R1"]["mean"], color="k", ls=":", lw=1, label="X-Pool (all pairs)")
    ax.set_xscale("log"); ax.set_xlabel("X-Pool pair evaluations per query (log)"); ax.set_ylabel("t2v R@1 (%)")
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "r1_vs_budget.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    al = [r["alpha"] for r in S["alpha_sweep"]]
    ax.errorbar(al, [r["R1"]["mean"] for r in S["alpha_sweep"]], yerr=[r["R1"]["std"] for r in S["alpha_sweep"]],
                marker="o", capsize=2, label="X-Pool head, logits × alpha")
    ax.axhline(S["scoring"]["mean"]["R1"]["mean"], color="gray", ls=":", label="mean pooling")
    ax.set_xlabel("alpha (0 = uniform attention = U-Pool, 1 = X-Pool)"); ax.set_ylabel("t2v R@1 (%)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "r1_vs_alpha.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--fixed_P", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50, 100])
    p.add_argument("--budgets", type=float, nargs="+", default=[2, 3, 5, 10, 20])
    p.add_argument("--adaptive", nargs="+", default=["0:50", "5:50"], help="P_lo:P_hi pairs")
    p.add_argument("--gallery_sizes", type=int, nargs="+", default=[1000, 10000])
    p.add_argument("--timing_P", type=int, nargs="+", default=[5, 20])
    p.add_argument("--timing_budget", type=float, default=5)
    p.add_argument("--skip_timing", action="store_true")
    a = p.parse_args()
    a.adaptive = [tuple(int(x) for x in s.split(":")) for s in a.adaptive]
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for i, r in enumerate(a.runs):
        R = analyze_run(r, a if i == 0 else argparse.Namespace(**{**vars(a), "skip_timing": True}), dev)
        json.dump(R, open(os.path.join(a.out_dir, f"{R['run']}.json"), "w"), indent=2)
        print(f"[{R['run']}] R@1 mean {R['scoring']['mean']['R1']:.1f} | U-Pool {R['scoring']['upool']['R1']:.1f} | "
              f"X-Pool {R['scoring']['xpool']['R1']:.1f} | sanity {R['sanity_alpha1_vs_dump_max_abs_diff']:.1e}", flush=True)
        runs.append(R)
    S = summarize(runs)
    json.dump(S, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    open(os.path.join(a.out_dir, "pooling_budget_report.md"), "w").write(to_md(S))
    plots(S, a.out_dir)
    print(to_md(S))


if __name__ == "__main__":
    main()
