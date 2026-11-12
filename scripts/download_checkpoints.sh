#!/usr/bin/env bash
# Official X-Pool checkpoints (links from the authors' README, layer6ai-labs/xpool).
# Usage: bash scripts/download_checkpoints.sh <dest_dir> [--with-7k]
#   Each file is ~1.8 GB (it also stores optimizer state). The 7k one is only needed for the
#   fallback plan (docs/ROADMAP.md §4), so it is skipped unless --with-7k is given.
set -euo pipefail
DEST="${1:-/content/drive/MyDrive/컴비 프로젝트/checkpoints}"
mkdir -p "${DEST}/official_9k"
# MSR-VTT-9k (paper: t2v R@1 46.9)
[ -f "${DEST}/official_9k/model_best.pth" ] || gdown --fuzzy "https://drive.google.com/file/d/1M2Y41B3a3AxzSJn-n-Xh0Edds97NU1ND/view?usp=sharing" -O "${DEST}/official_9k/model_best.pth"
# MSR-VTT-7k (paper: t2v R@1 43.9)
if [ "${2:-}" = "--with-7k" ]; then
  mkdir -p "${DEST}/official_7k"
  [ -f "${DEST}/official_7k/model_best.pth" ] || gdown --fuzzy "https://drive.google.com/file/d/1KW6VQiiTHpfMcK8GIgRq-5aWAgf7rGPj/view?usp=sharing" -O "${DEST}/official_7k/model_best.pth"
fi
ls -la "${DEST}"/official_*/
