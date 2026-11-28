"""Learning curves (test R@1 at every evaluation) for several runs, from eval_history.jsonl.

  python analysis/plot_curves.py --runs xpool_9k_s24 xpool_9k_s0 xpool_9k_s1 \
      --out experiments/results/figures/learning_curves.png
Note: seed 24 ran in 3 Colab sessions; evaluations repeated after a resume are de-duplicated by
global_step (the later one is kept; resumed runs were verified identical, see sessions_reconstructed.json).
"""
import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")


def load_curve(run, root=ROOT):
    by_step = {}
    for line in open(os.path.join(root, run, "eval_history.jsonl")):
        if line.strip():
            r = json.loads(line)
            if "R1" in r and "global_step" in r:
                by_step[int(r["global_step"])] = r
    steps = sorted(by_step)
    return steps, [by_step[s]["R1"] for s in steps], [by_step[s].get("R1-window") for s in steps]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--paper", type=float, default=46.9)
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.4))
    summary = {}
    for i, r in enumerate(a.runs):
        s, r1, w = load_curve(r)
        ax.plot(s, r1, color=f"C{i}", lw=1, alpha=0.45)
        ax.plot(s, w, color=f"C{i}", lw=1.8, label=f"{r} (5-eval window)")
        summary[r] = {"n_evals": len(s), "max_single_R1": max(r1), "last_R1": r1[-1],
                      "max_window_R1": max(x for x in w if x is not None)}
    ax.axhline(a.paper, color="k", ls=":", lw=1, label=f"paper {a.paper}")
    for e in range(1, 5):
        ax.axvline(5625 * e, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel("training step (epoch boundaries every 5,625 steps)")
    ax.set_ylabel("1k-A t2v R@1 (%)")
    ax.set_ylim(38, 48.5)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=160)
    json.dump(summary, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
