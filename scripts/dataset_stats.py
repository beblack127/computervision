"""Compute MSR-VTT statistics for the split protocol used by X-Pool, directly from the
annotation files shipped with the official code (third_party/xpool/data/MSRVTT).

  python scripts/dataset_stats.py --out_json experiments/results/dataset_stats.json \
      --out_md docs/DATASET_FACTSHEET_TABLES.md
"""
import argparse
import collections
import json
import os
import re
import statistics as st

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "third_party", "xpool", "data", "MSRVTT")

TEMPORAL_WORDS = ["then", "after", "before", "while", "begins", "starts", "start", "finally", "first",
                  "again", "later", "until", "next", "end", "ends"]
STOP = set("a an the is are of and in on to with at for his her their its this that it while from by "
           "as be being into out up down over some two one man woman person people men women".split())


def official_split(vid):
    """MSR-VTT 2016 release: train video0-6512, validate 6513-7009, test 7010-9999.
    (The 'split' field in this MSRVTT_data.json is 'train' for every video, so we use id ranges.)"""
    n = int(vid.replace("video", ""))
    return "train" if n <= 6512 else ("validate" if n <= 7009 else "test")


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def describe(xs):
    return {"mean": round(st.mean(xs), 2), "median": st.median(xs), "min": min(xs), "max": max(xs),
            "std": round(st.pstdev(xs), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DATA)
    p.add_argument("--out_json", default=os.path.join(ROOT, "experiments", "results", "dataset_stats.json"))
    p.add_argument("--out_md", default=os.path.join(ROOT, "docs", "DATASET_FACTSHEET_TABLES.md"))
    a = p.parse_args()

    db = json.load(open(os.path.join(a.data_dir, "MSRVTT_data.json")))
    videos = {v["video_id"]: v for v in db["videos"]}
    caps = collections.defaultdict(list)
    for s in db["sentences"]:
        caps[s["video_id"]].append(s["caption"])
    tr9 = pd.read_csv(os.path.join(a.data_dir, "MSRVTT_train.9k.csv"))["video_id"].tolist()
    tr7 = pd.read_csv(os.path.join(a.data_dir, "MSRVTT_train.7k.csv"))["video_id"].tolist()
    te = pd.read_csv(os.path.join(a.data_dir, "MSRVTT_JSFUSION_test.csv"))

    out = {"source_files": sorted(os.listdir(a.data_dir))}
    out["full_dataset"] = {
        "n_videos": len(videos), "n_captions": len(db["sentences"]),
        "captions_per_video": describe([len(caps[v]) for v in videos]),
        "split_field_values_in_json": dict(collections.Counter(v["split"] for v in videos.values())),
        "official_split_counts_by_id_range": dict(collections.Counter(official_split(v) for v in videos)),
        "clip_duration_sec": describe([round(v["end time"] - v["start time"], 2) for v in videos.values()]),
        "total_hours": round(sum(v["end time"] - v["start time"] for v in videos.values()) / 3600, 2),
        "n_categories": len({v["category"] for v in videos.values()}),
    }
    test_vids = te["video_id"].tolist()
    out["protocol"] = {
        "train_9k": {"n_videos": len(set(tr9)), "n_train_pairs": sum(len(caps[v]) for v in tr9),
                     "from_official_splits": dict(collections.Counter(official_split(v) for v in tr9))},
        "train_7k": {"n_videos": len(set(tr7)), "n_train_pairs": sum(len(caps[v]) for v in tr7),
                     "from_official_splits": dict(collections.Counter(official_split(v) for v in tr7))},
        "test_1kA": {"n_pairs": len(te), "n_videos": te["video_id"].nunique(),
                     "from_official_splits": dict(collections.Counter(official_split(v) for v in test_vids)),
                     "query_in_MSRVTT_captions": int(sum(r.sentence in caps[r.video_id] for r in te.itertuples()))},
        "overlap_train9k_test1kA": len(set(tr9) & set(test_vids)),
        "overlap_train7k_test1kA": len(set(tr7) & set(test_vids)),
        "unused_videos_in_9k_protocol": len(set(videos) - set(tr9) - set(test_vids)),
        "steps_per_epoch_bs32_9k": -(-sum(len(caps[v]) for v in tr9) // 32),
    }
    train_caps = [c for v in tr9 for c in caps[v]]
    test_caps = te["sentence"].tolist()
    out["caption_length_words"] = {"train_9k": describe([len(words(c)) for c in train_caps]),
                                   "test_1kA": describe([len(words(c)) for c in test_caps])}
    out["vocab_size"] = {"train_9k": len({w for c in train_caps for w in words(c)}),
                         "test_1kA": len({w for c in test_caps for w in words(c)})}
    tw = {w: int(sum(w in words(c) for c in test_caps)) for w in TEMPORAL_WORDS}
    out["test_queries_with_temporal_words"] = {"per_word": {k: v for k, v in tw.items() if v},
                                               "any": int(sum(any(w in words(c) for w in TEMPORAL_WORDS)
                                                              for c in test_caps))}
    # category distribution + data-driven description (top words; MSR-VTT json has ids only)
    cat_tr = collections.Counter(videos[v]["category"] for v in tr9)
    cat_te = collections.Counter(videos[v]["category"] for v in test_vids)
    cat_words = collections.defaultdict(collections.Counter)
    for v in tr9:
        for c in caps[v]:
            cat_words[videos[v]["category"]].update(w for w in words(c) if w not in STOP and len(w) > 2)
    out["categories"] = {int(k): {"train_9k_videos": cat_tr[k], "test_1kA_videos": cat_te.get(k, 0),
                                  "top_words": [w for w, _ in cat_words[k].most_common(6)]}
                         for k in sorted(cat_tr)}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=2)

    # ---- markdown tables (numbers only; the report text must be written by the student) ----
    f = out["full_dataset"]; pr = out["protocol"]
    L = ["# MSR-VTT — computed statistics (auto-generated by scripts/dataset_stats.py)", "",
         "Numbers computed from the annotation files shipped with the official X-Pool code. "
         "Regenerate instead of editing by hand.", "",
         "## Whole dataset", "", "| item | value |", "|---|---|",
         f"| videos | {f['n_videos']:,} |", f"| captions | {f['n_captions']:,} |",
         f"| captions / video | {f['captions_per_video']['min']}–{f['captions_per_video']['max']} (mean {f['captions_per_video']['mean']}) |",
         f"| clip duration (s) | mean {f['clip_duration_sec']['mean']}, median {f['clip_duration_sec']['median']}, range {f['clip_duration_sec']['min']}–{f['clip_duration_sec']['max']} |",
         f"| total duration | {f['total_hours']} h |",
         f"| official 2016 splits by id range (train/validate/test) | {f['official_split_counts_by_id_range']} |",
         f"| `split` field inside this json | {f['split_field_values_in_json']} (not informative) |",
         f"| categories | {f['n_categories']} (ids only in the json) |", "",
         "## Retrieval protocol used by X-Pool (and most CLIP-based work)", "",
         "| split | videos | text-video pairs | origin (official 2016 split, by id range) |", "|---|---|---|---|",
         f"| train 9k (main) | {pr['train_9k']['n_videos']:,} | {pr['train_9k']['n_train_pairs']:,} | {pr['train_9k']['from_official_splits']} |",
         f"| train 7k (alt.) | {pr['train_7k']['n_videos']:,} | {pr['train_7k']['n_train_pairs']:,} | {pr['train_7k']['from_official_splits']} |",
         f"| test 1k-A | {pr['test_1kA']['n_videos']:,} | {pr['test_1kA']['n_pairs']:,} (1 query/video) | {pr['test_1kA']['from_official_splits']} |",
         "", f"- train9k ∩ test1k-A = {pr['overlap_train9k_test1kA']} videos; "
             f"videos unused by the 9k protocol = {pr['unused_videos_in_9k_protocol']}",
         f"- 1k-A queries that are verbatim MSR-VTT captions of their video: {pr['test_1kA']['query_in_MSRVTT_captions']}/1000",
         f"- steps/epoch at batch 32 (9k): {pr['steps_per_epoch_bs32_9k']:,} → 5 epochs = {5 * pr['steps_per_epoch_bs32_9k']:,} steps",
         "- **No validation split exists in this protocol**: the official code selects `model_best.pth` on the 1k-A test set "
         "(see docs/BASELINE_XPOOL.md).", "",
         "## Text", "", "| | train 9k captions | test 1k-A queries |", "|---|---|---|",
         f"| words / caption (mean, median, max) | {out['caption_length_words']['train_9k']['mean']}, {out['caption_length_words']['train_9k']['median']}, {out['caption_length_words']['train_9k']['max']} "
         f"| {out['caption_length_words']['test_1kA']['mean']}, {out['caption_length_words']['test_1kA']['median']}, {out['caption_length_words']['test_1kA']['max']} |",
         f"| vocabulary | {out['vocab_size']['train_9k']:,} | {out['vocab_size']['test_1kA']:,} |",
         f"| queries with a temporal word ({', '.join(TEMPORAL_WORDS)}) | – | {out['test_queries_with_temporal_words']['any']} |",
         "", "## Categories (id → videos; top caption words as a data-driven hint of the topic)", "",
         "| id | train 9k | test 1k-A | top words |", "|---|---|---|---|"]
    for k, v in out["categories"].items():
        L.append(f"| {k} | {v['train_9k_videos']} | {v['test_1kA_videos']} | {', '.join(v['top_words'])} |")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print(json.dumps(out["protocol"], indent=2))


if __name__ == "__main__":
    main()
