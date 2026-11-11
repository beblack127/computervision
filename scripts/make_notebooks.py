"""Generate the Colab notebooks in notebooks/ (keeps them reviewable as code).

  python scripts/make_notebooks.py
"""
import os

import nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "notebooks")

SETUP = r'''# ==== 0. Settings (edit if needed) ====
DRIVE_ROOT = "/content/drive/MyDrive/cv_hw2"      # persistent: archive copy, checkpoints, runs
REPO_URL   = "https://github.com/beblack127/computervision.git"
BRANCH     = "main"
REPO       = "/content/computervision"
LOCAL_DATA = "/content/data/msrvtt"                # fast local disk (wiped when runtime ends)
CACHE_DIR  = "/content/cache/msrvtt_u12"           # frame cache (uniform 12 frames, raw uint8)
NUM_WORKERS = 8                                    # official default; keep fixed across runs

from google.colab import drive
drive.mount("/content/drive")
import os, subprocess, json, time
os.makedirs(DRIVE_ROOT, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!nproc && free -g | head -2 && df -h /content | tail -1
'''

CLONE = r'''# ==== 1. Code: repo + pinned official X-Pool + deps ====
if not os.path.isdir(REPO):
    !git clone -b {BRANCH} {REPO_URL} {REPO}
else:
    !git -C {REPO} pull --ff-only
%cd {REPO}
!pip -q install -r requirements-colab.txt
!bash scripts/setup_xpool.sh
import sys; sys.path.insert(0, f"{REPO}/src")
!PYTHONPATH=src python -m pytest -q tests/ 2>&1 | tail -3
'''

DATA = r'''# ==== 2. MSR-VTT videos (first time: download ~6.5GB and copy to Drive; later: copy from Drive) ====
t = time.time()
!python scripts/prepare_msrvtt.py --drive_dir {DRIVE_ROOT}/data --local_dir {LOCAL_DATA}
print(f"data ready in {time.time()-t:.0f}s")
'''

CACHE = r'''# ==== 3. Frame cache (bit-identical to official decoding) + verification on real videos ====
t = time.time()
!PYTHONPATH=src python -m tvr.frame_cache build --videos_dir {LOCAL_DATA}/videos --cache_dir {CACHE_DIR}
!PYTHONPATH=src python -m tvr.frame_cache stats --cache_dir {CACHE_DIR}
!PYTHONPATH=src python -m tvr.frame_cache verify --videos_dir {LOCAL_DATA}/videos --cache_dir {CACHE_DIR} --n 50
print(f"cache step: {time.time()-t:.0f}s")
!du -sh {CACHE_DIR}
!df -h /content | tail -1
'''

COMMON = ('--huggingface --dataset_name MSRVTT --msrvtt_train_file 9k --cache_dir {CACHE_DIR} '
          '--num_workers {NUM_WORKERS} --batch_size 32')


def go_no_go():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 00 · Go/No-Go for X-Pool on Colab\n"
            "Runtime: **A100, High-RAM**. Purpose: measure, before the proposal (11/15), whether a faithful "
            "full-scale X-Pool reproduction fits our compute.\n\n"
            "**Decision rule** (also written to `go_no_go.json`):\n"
            "- official 9k checkpoint re-evaluated with our pipeline: |R@1 − 46.9| ≤ 1.0, and\n"
            "- estimated full training run (5 epochs, 50 evals) ≤ 6 h on this GPU\n\n"
            "→ **GO** with full MSR-VTT-9k. Otherwise see docs/ROADMAP.md §Fallbacks (7k split / plan B).\n\n"
            "Expected wall time: ~1–1.5 h the first time (download dominates)."),
        nbf.v4.new_code_cell(SETUP),
        nbf.v4.new_code_cell(CLONE),
        nbf.v4.new_code_cell(DATA),
        nbf.v4.new_code_cell(CACHE),
        nbf.v4.new_code_cell(
            "# ==== 4. Official checkpoint -> our evaluation pipeline (expect t2v R@1 ≈ 46.9) ====\n"
            "!bash scripts/download_checkpoints.sh {DRIVE_ROOT}/checkpoints\n"
            "!PYTHONPATH=src python -m tvr.evaluate --exp_name official_9k_eval " + COMMON +
            " --output_dir {DRIVE_ROOT}/runs --ckpt {DRIVE_ROOT}/checkpoints/official_9k/model_best.pth"
            " --dump_dir {DRIVE_ROOT}/runs/official_9k_eval/dump\n"),
        nbf.v4.new_code_cell(
            "# ==== 5. Speed benchmark: 200 real training steps (no evals) + 1 eval ====\n"
            "!PYTHONPATH=src python -m tvr.train --exp_name bench_9k " + COMMON +
            " --noclip_lr 3e-5 --transformer_dropout 0.3 --output_dir {DRIVE_ROOT}/runs"
            " --max_steps 200 --evals_per_epoch 0 --no_tensorboard\n"),
        nbf.v4.new_code_cell(
            "# ==== 6. Summary + decision ====\n"
            "ev = json.load(open(f'{DRIVE_ROOT}/runs/official_9k_eval/dump/eval_original.json'))\n"
            "bm = json.load(open(f'{DRIVE_ROOT}/runs/bench_9k/run_meta.json'))\n"
            "prep = json.load(open(f'{LOCAL_DATA}/prepare_report.json'))\n"
            "cache = json.load(open(f'{CACHE_DIR}/index.json'))\n"
            "r1 = ev['t2v']['R1']; hours = bm['benchmark']['estimated_full_run_hours']\n"
            "summary = {\n"
            "  'gpu': bm['env'].get('gpu_name'), 'gpu_mem_GB': bm['env'].get('gpu_total_memory_GB'),\n"
            "  'cpu_count': bm['env'].get('cpu_count'), 'ram_GB': bm['env'].get('ram_total_GB'),\n"
            "  'data_source': prep['source'], 'n_videos': prep['n_unique_ids'], 'n_missing': prep['n_missing'],\n"
            "  'cache_build_seconds': cache['build_seconds'], 'cache_GB': round(cache['total_bytes_raw']/1e9, 1),\n"
            "  'official_ckpt_t2v': ev['t2v'], 'official_ckpt_v2t': ev['v2t'], 'eval_seconds': ev['eval_seconds'],\n"
            "  'benchmark': bm['benchmark'], 'peak_gpu_mem_GB': bm.get('peak_gpu_mem_reserved_GB'),\n"
            "  'versions': {k: bm['env'].get(k) for k in ['torch','torchvision','transformers','opencv','python']},\n"
            "}\n"
            "summary['GO'] = bool(abs(r1 - 46.9) <= 1.0 and hours <= 6)\n"
            "os.makedirs(f'{REPO}/experiments/results', exist_ok=True)\n"
            "for path in [f'{DRIVE_ROOT}/go_no_go.json', f'{REPO}/experiments/results/go_no_go.json']:\n"
            "    json.dump(summary, open(path, 'w'), indent=2, default=float)\n"
            "print(json.dumps(summary, indent=2, default=float))\n"
            "print('\\nDECISION:', 'GO ✅' if summary['GO'] else 'NO-GO ⚠️  -> see docs/ROADMAP.md Fallbacks')\n"),
        nbf.v4.new_markdown_cell(
            "## After running\n"
            "1. Download `experiments/results/go_no_go.json` (or copy from Drive `cv_hw2/go_no_go.json`) and commit it "
            "in VSCode (`experiments/results/go_no_go.json`).\n"
            "2. Paste the printed summary into `docs/HANDOFF.md` (Status section) so Claude/Codex can plan the next step.\n"
            "3. These numbers (GPU, memory, timings) feed the Proposal's *Dataset & compute* section."),
    ]
    return nb


def train_baseline():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 01 · Train X-Pool baseline (MSR-VTT 9k, official recipe)\n"
            "One run per seed (planned seeds: 24 = official default, 0, 1). ~ see go_no_go.json for time.\n"
            "If Colab disconnects, just re-run all cells: training resumes from the last finished epoch "
            "(`resume_state.pth` on Drive)."),
        nbf.v4.new_code_cell(SETUP + "\nSEED = 24   # <- change per run\nEXP = f\"xpool_9k_s{SEED}\"\n"),
        nbf.v4.new_code_cell(CLONE),
        nbf.v4.new_code_cell(DATA),
        nbf.v4.new_code_cell(CACHE),
        nbf.v4.new_code_cell(
            "# ==== 4. Train (official 9k hyper-parameters) ====\n"
            "RUN_DIR = f'{DRIVE_ROOT}/runs/{EXP}'; os.makedirs(RUN_DIR, exist_ok=True)\n"
            "resume = f'--resume_state {RUN_DIR}/resume_state.pth' if os.path.exists(f'{RUN_DIR}/resume_state.pth') else ''\n"
            "print('resume:', bool(resume))\n"
            "!PYTHONPATH=src python -m tvr.train --exp_name {EXP} --seed {SEED} " + COMMON +
            " --noclip_lr 3e-5 --transformer_dropout 0.3 --output_dir {DRIVE_ROOT}/runs"
            " --tb_log_dir {DRIVE_ROOT}/tb {resume} 2>&1 | tee -a {RUN_DIR}/train.log | grep -E 'Val Epoch|R@1|Best|Saving|resume|Error|Traceback'\n"),
        nbf.v4.new_code_cell(
            "# ==== 5. Evaluate best (official protocol) and last checkpoints, with analysis dumps ====\n"
            "for ck in ['model_best', 'model_last']:\n"
            "    !PYTHONPATH=src python -m tvr.evaluate --exp_name {EXP}_eval_{ck} " + COMMON +
            " --output_dir {DRIVE_ROOT}/runs --ckpt {RUN_DIR}/{ck}.pth --dump_dir {RUN_DIR}/dump_{ck}\n"
            "# temporal-sensitivity probes on the official-protocol checkpoint\n"
            "for order in ['reversed', 'shuffled', 'middle']:\n"
            "    !PYTHONPATH=src python -m tvr.evaluate --exp_name {EXP}_eval_probe " + COMMON +
            " --output_dir {DRIVE_ROOT}/runs --ckpt {RUN_DIR}/model_best.pth --dump_dir {RUN_DIR}/dump_model_best"
            " --frame_order {order}\n"),
        nbf.v4.new_code_cell(
            "# ==== 6. Collect small result files into the repo (commit them from VSCode) ====\n"
            "dst = f'{REPO}/experiments/results/{EXP}'; os.makedirs(dst, exist_ok=True)\n"
            "!cp {RUN_DIR}/run_meta.json {RUN_DIR}/eval_history.jsonl {dst}/ 2>/dev/null; "
            "cp {RUN_DIR}/dump_model_best/eval_*.json {dst}/ 2>/dev/null; "
            "cp {RUN_DIR}/dump_model_last/eval_original.json {dst}/eval_last_original.json 2>/dev/null; ls {dst}\n"),
    ]
    return nb


def analysis():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# 02 · Limitation analysis on a trained run (CPU runtime is enough)\n"
            "Inputs: `dump_model_best/dump_*.pt` from notebook 01 and the frame cache (only for thumbnails)."),
        nbf.v4.new_code_cell(SETUP + "\nEXP = 'xpool_9k_s24'\nRUN_DIR = f'{DRIVE_ROOT}/runs/{EXP}'\n"),
        nbf.v4.new_code_cell(CLONE),
        nbf.v4.new_code_cell(
            "# ==== Analysis report: per-category, query length, temporal words, hubness, frame-order probes ====\n"
            "!PYTHONPATH=src python analysis/analyze_dump.py --dump_dir {RUN_DIR}/dump_model_best "
            "--data_dir third_party/xpool/data/MSRVTT --out_dir {RUN_DIR}/analysis\n"
            "print(open(f'{RUN_DIR}/analysis/analysis_report.md').read())\n"),
        nbf.v4.new_code_cell(
            "# ==== Failure-case gallery and attention over frames (needs the frame cache for thumbnails) ====\n"
            "# Rebuild cache only if missing (CPU runtime: ~slower but fine)\n"
            "if not os.path.exists(f'{CACHE_DIR}/index.json'):\n"
            "    !python scripts/prepare_msrvtt.py --drive_dir {DRIVE_ROOT}/data --local_dir {LOCAL_DATA}\n"
            "    !PYTHONPATH=src python -m tvr.frame_cache build --videos_dir {LOCAL_DATA}/videos --cache_dir {CACHE_DIR}\n"
            "!PYTHONPATH=src python analysis/failure_gallery.py --dump {RUN_DIR}/dump_model_best/dump_original.pt "
            "--ckpt {RUN_DIR}/model_best.pth --cache_dir {CACHE_DIR} --out_dir {RUN_DIR}/analysis/gallery --n 12\n"),
    ]
    return nb


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("00_go_no_go.ipynb", go_no_go), ("01_train_baseline.ipynb", train_baseline),
                     ("02_analysis.ipynb", analysis)]:
        nb = fn()
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["colab"] = {"provenance": [], "gpuType": "A100", "machine_shape": "hm"}
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
        nbf.write(nb, os.path.join(OUT, name))
        print("wrote", name)


if __name__ == "__main__":
    main()
