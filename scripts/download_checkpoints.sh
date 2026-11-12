#!/usr/bin/env bash
# Official X-Pool checkpoints (links from the authors' README, layer6ai-labs/xpool).
# Usage: bash scripts/download_checkpoints.sh <dest_dir>
set -euo pipefail
DEST="${1:-/content/drive/MyDrive/컴비 프로젝트/checkpoints}"
mkdir -p "${DEST}/official_9k" "${DEST}/official_7k"
# MSR-VTT-9k (paper: t2v R@1 46.9)
[ -f "${DEST}/official_9k/model_best.pth" ] || gdown --fuzzy "https://drive.google.com/file/d/1M2Y41B3a3AxzSJn-n-Xh0Edds97NU1ND/view?usp=sharing" -O "${DEST}/official_9k/model_best.pth"
# MSR-VTT-7k (paper: t2v R@1 43.9)
[ -f "${DEST}/official_7k/model_best.pth" ] || gdown --fuzzy "https://drive.google.com/file/d/1KW6VQiiTHpfMcK8GIgRq-5aWAgf7rGPj/view?usp=sharing" -O "${DEST}/official_7k/model_best.pth"
ls -la "${DEST}"/official_*/
