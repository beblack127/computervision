"""Limitation analysis of an X-Pool run from the dumps written by `tvr.evaluate --dump_dir`.

  python analysis/analyze_dump.py --dump_dir RUN/dump_model_best \
      --data_dir third_party/xpool/data/MSRVTT --out_dir RUN/analysis

Produces analysis.json, analysis_report.md (tables) and a few PNG plots:
  1. sanity: metrics recomputed from the dumped similarity matrix
  2. per-category / per-query-length / temporal-word breakdown (where does it fail?)
  3. frame-order probes (does the model use temporal order at all?)
  4. hubness (are a few videos retrieved for many queries?)
  5. failure taxonomy: is the top-1 wrong video's own caption nearly identical to the query?
     (= annotation ambiguity) vs. genuinely different content (= model error)
  6. DIAGNOSTIC ONLY: test-time re-normalisation (dual-softmax) that uses the whole test query
     set -- reported to quantify hubness, NOT a method we may claim (it peeks at test queries).
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import torch

TEMPORAL = {"then", "after", "before", "while", "begins", "starts", "start", "finally", "first",
            "again", "later", "until", "next", "end", "ends"}


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def ranks_from_sims(sims):
    """sims[i, j] = similarity(query i, video j); ground truth of query i is video i."""
    order = torch.argsort(sims, dim=1, descending=True)
    gt = torch.arange(sims.shape[0])[:, None]
    return (order == gt).float().argmax(dim=1).numpy(), order.numpy()


def metrics(r):
    r = np.asarray(r)
    return {"n": int(len(r)), "R1": float(100 * np.mean(r == 0)), "R5": float(100 * np.mean(r < 5)),
            "R10": float(100 * np.mean(r < 10)), "MdR": float(np.median(r) + 1), "MnR": float(np.mean(r) + 1)}


def dual_softmax(sims, temp=100.0):
    """Dual-softmax re-weighting over the query axis (DSL-style). Diagnostic only."""
    prior = torch.softmax(sims * temp, dim=0)
    return sims * prior


def hubness(order, k=10):
    n = order.shape[0]
    top1 = collections.Counter(order[:, 0].tolist())
    topk = collections.Counter(order[:, :k].reshape(-1).tolist())
    nk = np.array([topk.get(j, 0) for j in range(n)])
    skew = float(((nk - nk.mean()) ** 3).mean() / (nk.std() ** 3 + 1e-9))
    return {"k": k, "N_k_skewness": skew, "max_N1": int(max(top1.values())),
            "videos_never_top1": int(n - len(top1)), "videos_never_topk": int((nk == 0).sum()),
            "top_hubs": [(int(j), int(c)) for j, c in top1.most_common(5)]}, nk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_fail", type=int, default=25)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = torch.load(os.path.join(a.dump_dir, "dump_original.pt"), weights_only=False)
    sims, texts, vids = d["sims"].float(), d["texts"], d["video_ids"]
    assert list(d["query_video_ids"]) == list(vids), "expects 1k-A layout (one query per video, same order)"
    db = json.load(open(os.path.join(a.data_dir, "MSRVTT_data.json")))
    cat = {v["video_id"]: v["category"] for v in db["videos"]}
    ranks, order = ranks_from_sims(sims)
    R = {"overall": metrics(ranks)}

    # 2. breakdowns
    by_cat = collections.defaultdict(list)
    by_len = collections.defaultdict(list)
    by_tmp = collections.defaultdict(list)
    for i, (t, v) in enumerate(zip(texts, vids)):
        by_cat[cat[v]].append(ranks[i])
        n = len(words(t))
        by_len["<=6" if n <= 6 else "7-9" if n <= 9 else "10-12" if n <= 12 else ">=13"].append(ranks[i])
        by_tmp["temporal word" if TEMPORAL & set(words(t)) else "no temporal word"].append(ranks[i])
    R["per_category"] = {int(k): metrics(v) for k, v in sorted(by_cat.items())}
    R["per_query_length"] = {k: metrics(by_len[k]) for k in ["<=6", "7-9", "10-12", ">=13"] if by_len[k]}
    R["temporal_words"] = {k: metrics(v) for k, v in by_tmp.items()}

    # 3. frame-order probes (eval_*.json written by tvr.evaluate --frame_order)
    probes = {}
    for order_name in ["original", "reversed", "shuffled", "middle"]:
        f = os.path.join(a.dump_dir, f"eval_{order_name}.json")
        if os.path.exists(f):
            probes[order_name] = json.load(open(f))["t2v"]
    R["frame_order_probes"] = probes

    # 4. hubness + 6. diagnostic dual-softmax
    R["hubness"], nk = hubness(order)
    ds_ranks, _ = ranks_from_sims(dual_softmax(sims))
    R["diagnostic_dual_softmax_uses_test_queries"] = metrics(ds_ranks)

    # 5. failure taxonomy via text-text similarity between the query and the caption of the
    #    wrongly retrieved video (1k-A has exactly one caption per test video)
    te = d["text_embeds"].float()
    te = te / te.norm(dim=-1, keepdim=True)
    fails = np.where(ranks > 0)[0]
    tt = (te[fails] * te[order[fails, 0]]).sum(-1).numpy()
    R["failures"] = {"n": int(len(fails)), "query_vs_wrong_caption_cos": {
        "mean": float(tt.mean()), "p50": float(np.median(tt)),
        "share_cos_ge_0.9": float((tt >= 0.9).mean()), "share_cos_ge_0.95": float((tt >= 0.95).mean())}}
    worst = sorted(fails, key=lambda i: -ranks[i])[: a.n_fail]
    near = [i for i in sorted(fails, key=lambda i: -float((te[i] * te[order[i, 0]]).sum())) if ranks[i] > 0][: a.n_fail]
    R["worst_queries"] = [{"query": texts[i], "gt": vids[i], "rank": int(ranks[i]) + 1,
                           "top1": vids[order[i, 0]], "top1_caption": texts[order[i, 0]], "category": cat[vids[i]]}
                          for i in worst]
    R["ambiguous_failures"] = [{"query": texts[i], "gt": vids[i], "rank": int(ranks[i]) + 1,
                                "top1": vids[order[i, 0]], "top1_caption": texts[order[i, 0]]} for i in near]
    json.dump(R, open(os.path.join(a.out_dir, "analysis.json"), "w"), indent=2, default=float)

    # ---- plots ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(R["per_category"].keys())
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar([str(k) for k in ks], [R["per_category"][k]["R1"] for k in ks], color="#4C72B0")
        ax.axhline(R["overall"]["R1"], color="gray", ls="--", lw=1)
        ax.set_xlabel("MSR-VTT category id"); ax.set_ylabel("t2v R@1 (%)"); ax.set_title("R@1 by category (1k-A)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "r1_by_category.png"), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(np.minimum(ranks + 1, 101), bins=50, color="#4C72B0")
        ax.set_xlabel("rank of ground-truth video (clipped at 101)"); ax.set_ylabel("#queries")
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "rank_hist.png"), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(nk, bins=range(0, int(nk.max()) + 2), color="#DD8452")
        ax.set_xlabel("N_10 (#queries that put the video in their top-10)"); ax.set_ylabel("#videos")
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "hubness_n10.png"), dpi=150); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plotting skipped:", e)

    # ---- markdown report (tables only) ------------------------------------------------------
    def row(name, m):
        return f"| {name} | {m['n']} | {m['R1']:.1f} | {m['R5']:.1f} | {m['R10']:.1f} | {m['MdR']:.1f} | {m['MnR']:.1f} |"
    H = "| subset | n | R@1 | R@5 | R@10 | MdR | MnR |\n|---|---|---|---|---|---|---|"
    L = ["# Analysis report (auto-generated)", "", f"dump: `{a.dump_dir}`", "", "## Overall", "", H,
         row("all", R["overall"]), "", "## By query length (words)", "", H]
    L += [row(k, v) for k, v in R["per_query_length"].items()]
    L += ["", "## Temporal words in the query", "", H] + [row(k, v) for k, v in R["temporal_words"].items()]
    L += ["", "## By category", "", H] + [row(str(k), v) for k, v in R["per_category"].items()]
    if probes:
        L += ["", "## Frame-order probes (same checkpoint, frames permuted at test time)", "",
              "| frames | R@1 | R@5 | R@10 | MnR |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['R1']:.1f} | {v['R5']:.1f} | {v['R10']:.1f} | {v['MeanR']:.1f} |" for k, v in probes.items()]
    h = R["hubness"]
    L += ["", "## Hubness", "", f"- N_10 skewness: {h['N_k_skewness']:.2f}; max times a single video is top-1: {h['max_N1']}",
          f"- videos never retrieved at top-1: {h['videos_never_top1']} / {len(vids)}; never in any top-10: {h['videos_never_topk']}",
          f"- DIAGNOSTIC (uses all test queries, not a valid method): dual-softmax R@1 = {R['diagnostic_dual_softmax_uses_test_queries']['R1']:.1f}",
          "", "## Failures", "", f"- failed queries (rank>1): {R['failures']['n']}",
          f"- cos(query, caption of wrongly retrieved video): mean {R['failures']['query_vs_wrong_caption_cos']['mean']:.3f}; "
          f"share ≥0.9: {100 * R['failures']['query_vs_wrong_caption_cos']['share_cos_ge_0.9']:.1f}%",
          "", "### Worst-ranked queries", "", "| rank | query | GT | top-1 (its caption) |", "|---|---|---|---|"]
    L += [f"| {w['rank']} | {w['query']} | {w['gt']} | {w['top1']} ({w['top1_caption']}) |" for w in R["worst_queries"][:15]]
    L += ["", "### Failures whose wrong top-1 has an almost identical caption (annotation ambiguity)", "",
          "| rank | query | top-1 caption |", "|---|---|---|"]
    L += [f"| {w['rank']} | {w['query']} | {w['top1_caption']} |" for w in R["ambiguous_failures"][:15]]
    open(os.path.join(a.out_dir, "analysis_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
