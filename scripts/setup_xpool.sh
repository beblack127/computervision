#!/usr/bin/env bash
# Clone the official X-Pool code (CVPR 2022) at a pinned commit into third_party/xpool.
# We do NOT vendor or edit the official code: all fixes live in src/tvr/ and are applied
# by subclassing / wrapping, so the baseline stays byte-identical to the authors' release.
set -euo pipefail

XPOOL_REPO="https://github.com/layer6ai-labs/xpool.git"
XPOOL_COMMIT="bc88718199007d395136adc99a3e69adbee874a0"   # HEAD of master as of 2023-04-25
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/xpool"

if [ -d "${DEST}/.git" ]; then
  echo "[setup_xpool] already present: ${DEST}"
else
  mkdir -p "${ROOT}/third_party"
  git clone "${XPOOL_REPO}" "${DEST}"
fi

cd "${DEST}"
git fetch --quiet origin "${XPOOL_COMMIT}" 2>/dev/null || true
git checkout --quiet "${XPOOL_COMMIT}"
echo "[setup_xpool] X-Pool pinned at $(git rev-parse HEAD)"
