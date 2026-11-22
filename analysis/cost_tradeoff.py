"""Accuracy-vs-cost trade-off of query-dependent (X-Pool) vs query-agnostic (mean) pooling.

Motivated by the proposal feedback: "analyze retrieval failures and the trade-off between
query-dependent pooling and computational cost". Uses only the dumps written by
`tvr.evaluate --dump_dir` plus the trained pooling weights -- no extra training.

  python analysis/cost_tradeoff.py --dump RUN/dump_model_best/dump_original.pt \
      --ckpt RUN/model_best.pth --data_dir third_party/xpool/data/MSRVTT --out_dir RUN/analysis/cost

What it measures
  1. Accuracy of the SAME trained model with two scoring rules on 1k-A:
       - X-Pool (official, query-dependent): similarities from the dump
       - mean pooling (query-agnostic): average of the same 12 frame embeddings
     Caveat to state in the report: the encoder was trained for X-Pool pooling, so mean pooling
     here is "swap the pooling at test time", not a separately trained CLIP4Clip.
  2. Per-query gain/loss of query dependence (rank under X-Pool vs mean), broken down by
     category, query length, temporal words and *frame diversity* of the video (mean pairwise
     cosine distance between its 12 frame embeddings; X-Pool's paper claims the benefit grows
     with content diversity).
  3. Two-stage retrieval (X-Pool paper, appendix C): shortlist top-P by mean pooling, re-rank
     only those by X-Pool. Accuracy vs P together with the number of X-Pool pair evaluations.
  4. Wall-clock time and memory of the ranking stage (embeddings pre-computed) for growing
     gallery sizes: mean pooling vs full X-Pool vs two-stage. Uses the GPU when available.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from failure_gallery import load_pooling  # noqa: E402

TEMPORAL = {"then", "after", "before", "while", "begins", "starts", "start", "finally", "first",
            "again", "later", "until", "next", "end", "ends"}


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def ranks_of(sims):
    """sims [Q, V]; ground truth of query i is video i -> 0-based rank per query."""
    gt = sims.diagonal()[:, None]
    return (sims > gt).sum(1).numpy()  # ties resolved in favour of the ground truth


def metrics(r):
    r = np.asarray(r)
    return {"n": int(len(r)), "R1": float(100 * np.mean(r == 0)), "R5": float(100 * np.mean(r < 5)),
            "R10": float(100 * np.mean(r < 10)), "MdR": float(np.median(r) + 1), "MnR": float(np.mean(r) + 1)}


def mean_pool_sims(text, frames):
    t = F.normalize(text.float(), dim=-1)
    v = F.normalize(frames.float().mean(1), dim=-1)
    return t @ v.t()


@torch.no_grad()
def xpool_sims(pool, text, frames, chunk=128):
    """Official X-Pool scoring for every (query, video) pair: [Q, V]. Chunked over videos."""
    t = F.normalize(text, dim=-1)
    out = []
    for i in range(0, frames.shape[0], chunk):
        pooled = F.normalize(pool(text, frames[i:i + chunk]), dim=-1)  # [v, Q, D]
        out.append(torch.einsum("qd,vqd->qv", t, pooled))
    return torch.cat(out, 1)


def two_stage(s_mean, s_xpool, P):
    """Rank by X-Pool inside the mean-pooling top-P shortlist; everything else keeps mean order
    behind the shortlist. Returns 0-based GT ranks."""
    Q, V = s_mean.shape
    P = min(P, V)
    top = torch.topk(s_mean, P, dim=1).indices  # [Q, P]
    ranks = np.empty(Q, dtype=np.int64)
    order_mean = torch.argsort(s_mean, dim=1, descending=True)
    for q in range(Q):
        short = top[q]
        if (short == q).any():
            sx = s_xpool[q, short]
            ranks[q] = int((sx > s_xpool[q, q]).sum())
        else:  # GT not shortlisted: placed after the P re-ranked items, in mean order
            pos = int((order_mean[q] == q).nonzero()[0, 0])
            ranks[q] = pos
    return ranks


def frame_diversity(frames):
    f = F.normalize(frames.float(), dim=-1)
    g = f @ f.transpose(1, 2)  # [V, F, F]
    n = f.shape[1]
    off = (g.sum((1, 2)) - n) / (n * (n - 1))
    return (1 - off).numpy()  # mean pairwise cosine distance


def bucket_stats(name_of, rx, rm, keys=None):
    groups = {}
    for i, k in enumerate(name_of):
        groups.setdefault(k, []).append(i)
    out = {}
    for k in (keys or sorted(groups)):
        if k not in groups:
            continue
        idx = np.array(groups[k])
        out[str(k)] = {"n": int(len(idx)), "R1_xpool": float(100 * np.mean(rx[idx] == 0)),
                       "R1_mean": float(100 * np.mean(rm[idx] == 0)),
                       "gain_R1": float(100 * (np.mean(rx[idx] == 0) - np.mean(rm[idx] == 0))),
                       "median_rank_change": float(np.median(rm[idx] - rx[idx]))}
    return out


@torch.no_grad()
def time_it(fn, device, repeat=3):
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
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None
    return float(np.median(ts)), peak


@torch.no_grad()
def cost_scaling(pool, text, frames, device, gallery_sizes, P, n_queries, chunk):
    """Ranking-stage cost with pre-computed embeddings. Gallery is tiled from the 1k-A videos
    (cost does not depend on content). Memory: measured GPU peak (MB) or analytic tensor size."""
    pool = pool.to(device).eval()
    q = text[:n_queries].float().to(device)
    res = []
    for N in gallery_sizes:
        reps = math.ceil(N / frames.shape[0])
        fr = frames.float().repeat(reps, 1, 1)[:N].to(device)
        mean_vecs = F.normalize(fr.mean(1), dim=-1)  # pre-computable offline
        tq = F.normalize(q, dim=-1)
        t_mean, m_mean = time_it(lambda: tq @ mean_vecs.t(), device)
        t_full, m_full = time_it(lambda: xpool_sims(pool, q, fr, chunk), device)

        def stage2():
            top = torch.topk(tq @ mean_vecs.t(), min(P, N), dim=1).indices
            for i in range(q.shape[0]):  # X-Pool only on the shortlist of each query
                xpool_sims(pool, q[i:i + 1], fr[top[i]], chunk)
        t_two, m_two = time_it(stage2, device, repeat=1)
        res.append({
            "gallery": N, "queries": int(q.shape[0]),
            "mean_ms": 1e3 * t_mean, "xpool_ms": 1e3 * t_full, f"two_stage_P{P}_ms": 1e3 * t_two,
            "mean_peak_MB": m_mean, "xpool_peak_MB": m_full, "two_stage_peak_MB": m_two,
            "index_MB_mean": N * 512 * 4 / 1024 ** 2,           # what must be stored per video
            "index_MB_xpool": N * fr.shape[1] * 512 * 4 / 1024 ** 2,
            "xpool_pairs": int(q.shape[0] * N), "two_stage_pairs": int(q.shape[0] * min(P, N)),
        })
        print(f"[cost] gallery {N:>6}: mean {1e3 * t_mean:8.2f} ms | xpool {1e3 * t_full:9.1f} ms | "
              f"two-stage(P={P}) {1e3 * t_two:9.1f} ms", flush=True)
        del fr
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--P", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 200, 500, 1000])
    p.add_argument("--P_cost", type=int, default=100, help="shortlist size used in the timing table")
    p.add_argument("--gallery_sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000])
    p.add_argument("--cost_queries", type=int, default=1000)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--skip_timing", action="store_true")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = torch.load(a.dump, weights_only=False)
    assert list(d["query_video_ids"]) == list(d["video_ids"]), "expects 1k-A layout"
    texts, vids = d["texts"], d["video_ids"]
    text, frames = d["text_embeds"].float(), d["vid_embeds"].float()
    s_x = d["sims"].float()
    s_m = mean_pool_sims(text, frames)
    pool = load_pooling(a.ckpt)

    # sanity: our re-implementation of X-Pool scoring reproduces the dumped similarities
    k = min(64, len(vids))
    s_check = xpool_sims(pool, text[:k], frames)
    sanity = float((s_check - s_x[:k]).abs().max())

    rx, rm = ranks_of(s_x), ranks_of(s_m)
    R = {"device": str(dev), "sanity_max_abs_diff_xpool_sims": sanity,
         "accuracy": {"xpool": metrics(rx), "mean_pool_same_encoder": metrics(rm)}}
    both = int(np.sum((rx == 0) & (rm == 0)))
    gain = int(np.sum((rx == 0) & (rm > 0)))
    loss = int(np.sum((rx > 0) & (rm == 0)))
    R["per_query_top1"] = {"both_correct": both, "xpool_only": gain, "mean_only": loss,
                           "both_wrong": int(len(rx) - both - gain - loss),
                           "rank_improved": int(np.sum(rx < rm)), "rank_worse": int(np.sum(rx > rm)),
                           "rank_same": int(np.sum(rx == rm))}

    db = json.load(open(os.path.join(a.data_dir, "MSRVTT_data.json")))
    cat = {v["video_id"]: v["category"] for v in db["videos"]}
    lens = [len(words(t)) for t in texts]
    len_b = ["<=6" if n <= 6 else "7-9" if n <= 9 else "10-12" if n <= 12 else ">=13" for n in lens]
    tmp_b = ["temporal word" if TEMPORAL & set(words(t)) else "no temporal word" for t in texts]
    div = frame_diversity(frames)
    qs = np.quantile(div, [0.25, 0.5, 0.75])
    div_b = ["Q1 (most static)" if x <= qs[0] else "Q2" if x <= qs[1] else "Q3" if x <= qs[2]
             else "Q4 (most diverse)" for x in div]
    R["breakdown"] = {
        "frame_diversity_quartile": bucket_stats(div_b, rx, rm, ["Q1 (most static)", "Q2", "Q3", "Q4 (most diverse)"]),
        "query_length": bucket_stats(len_b, rx, rm, ["<=6", "7-9", "10-12", ">=13"]),
        "temporal_words": bucket_stats(tmp_b, rx, rm),
        "category": bucket_stats([cat[v] for v in vids], rx, rm),
    }
    R["frame_diversity_quantiles"] = [float(x) for x in qs]

    R["two_stage"] = []
    for P in a.P:
        if P > len(vids):
            continue
        m = metrics(two_stage(s_m, s_x, P))
        m.update({"P": P, "xpool_pair_fraction": P / len(vids)})
        R["two_stage"].append(m)

    if not a.skip_timing:
        R["cost_scaling"] = cost_scaling(pool, text, frames, dev, a.gallery_sizes, a.P_cost,
                                         min(a.cost_queries, len(vids)), a.chunk)
    json.dump(R, open(os.path.join(a.out_dir, "cost_tradeoff.json"), "w"), indent=2, default=float)

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ps = [x["P"] for x in R["two_stage"]]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(Ps, [x["R1"] for x in R["two_stage"]], "o-", label="two-stage (mean → X-Pool top-P)")
        ax.axhline(R["accuracy"]["xpool"]["R1"], color="C1", ls="--", lw=1, label="full X-Pool")
        ax.axhline(R["accuracy"]["mean_pool_same_encoder"]["R1"], color="gray", ls=":", lw=1, label="mean pooling")
        ax.set_xscale("log"); ax.set_xlabel("shortlist size P (X-Pool pairs per query)"); ax.set_ylabel("t2v R@1 (%)")
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(a.out_dir, "two_stage_r1_vs_P.png"), dpi=150); plt.close(fig)
        if "cost_scaling" in R:
            cs = R["cost_scaling"]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            g = [x["gallery"] for x in cs]
            ax.plot(g, [x["mean_ms"] for x in cs], "o-", label="mean pooling")
            ax.plot(g, [x["xpool_ms"] for x in cs], "o-", label="X-Pool (all pairs)")
            ax.plot(g, [x[f"two_stage_P{a.P_cost}_ms"] for x in cs], "o-", label=f"two-stage P={a.P_cost}")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("gallery size (videos)")
            ax.set_ylabel(f"ranking time for {cs[0]['queries']} queries (ms)"); ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "cost_scaling.png"), dpi=150); plt.close(fig)
        b = R["breakdown"]["frame_diversity_quartile"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ks = list(b)
        xs = np.arange(len(ks))
        ax.bar(xs - 0.2, [b[k]["R1_mean"] for k in ks], 0.4, label="mean pooling", color="gray")
        ax.bar(xs + 0.2, [b[k]["R1_xpool"] for k in ks], 0.4, label="X-Pool", color="C1")
        ax.set_xticks(xs); ax.set_xticklabels(ks, fontsize=7); ax.set_ylabel("t2v R@1 (%)")
        ax.set_title("by frame diversity of the ground-truth video", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "gain_by_diversity.png"), dpi=150); plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print("plotting skipped:", e)

    # ---- markdown (tables only) ----
    ax_, am = R["accuracy"]["xpool"], R["accuracy"]["mean_pool_same_encoder"]
    L = ["# Query-dependent pooling: accuracy vs cost (auto-generated)", "", f"dump: `{a.dump}`  ·  device: {dev}",
         f"sanity (re-computed X-Pool sims vs dump, max |diff|): {sanity:.2e}", "",
         "## Same trained model, two scoring rules (1k-A, t2v)", "",
         "| scoring | R@1 | R@5 | R@10 | MdR | MnR |", "|---|---|---|---|---|---|",
         f"| X-Pool (query-dependent) | {ax_['R1']:.1f} | {ax_['R5']:.1f} | {ax_['R10']:.1f} | {ax_['MdR']:.1f} | {ax_['MnR']:.1f} |",
         f"| mean pooling (query-agnostic) | {am['R1']:.1f} | {am['R5']:.1f} | {am['R10']:.1f} | {am['MdR']:.1f} | {am['MnR']:.1f} |",
         "", "Caveat: the encoder was trained with X-Pool pooling; mean pooling is swapped in at test time.", "",
         "## Per-query top-1 outcome", ""]
    pq = R["per_query_top1"]
    L += [f"- both correct {pq['both_correct']}, X-Pool only {pq['xpool_only']}, mean only {pq['mean_only']}, "
          f"both wrong {pq['both_wrong']}",
          f"- rank improved by X-Pool {pq['rank_improved']}, worse {pq['rank_worse']}, same {pq['rank_same']}", ""]
    for name, tab in R["breakdown"].items():
        L += [f"## Gain of query dependence by {name}", "",
              "| bucket | n | R@1 mean | R@1 X-Pool | gain (pp) | median rank change |", "|---|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['R1_mean']:.1f} | {v['R1_xpool']:.1f} | {v['gain_R1']:+.1f} | {v['median_rank_change']:+.1f} |"
              for k, v in tab.items()]
        L.append("")
    L += ["## Two-stage retrieval (mean-pooling shortlist → X-Pool re-rank)", "",
          "| P | X-Pool pairs / query | R@1 | R@5 | R@10 | MnR |", "|---|---|---|---|---|---|"]
    L += [f"| {x['P']} | {x['P']} ({100 * x['xpool_pair_fraction']:.1f}%) | {x['R1']:.1f} | {x['R5']:.1f} | {x['R10']:.1f} | {x['MnR']:.1f} |"
          for x in R["two_stage"]]
    if "cost_scaling" in R:
        L += ["", f"## Ranking-stage cost ({dev}, embeddings pre-computed, {R['cost_scaling'][0]['queries']} queries)", "",
              f"| gallery | mean ms | X-Pool ms | two-stage P={a.P_cost} ms | X-Pool peak MB | index MB mean / X-Pool |",
              "|---|---|---|---|---|---|"]
        for x in R["cost_scaling"]:
            pk = "–" if x["xpool_peak_MB"] is None else f"{x['xpool_peak_MB']:.0f}"
            L.append(f"| {x['gallery']} | {x['mean_ms']:.2f} | {x['xpool_ms']:.1f} | {x[f'two_stage_P{a.P_cost}_ms']:.1f} | "
                     f"{pk} | {x['index_MB_mean']:.1f} / {x['index_MB_xpool']:.1f} |")
    open(os.path.join(a.out_dir, "cost_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:14]))


if __name__ == "__main__":
    main()
