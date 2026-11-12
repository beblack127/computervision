# ITE4052 Computer Vision: Assignment 2

**Text-to-Video Retrieval: X-Pool(CVPR 2022) 재현 → 한계 분석 → 개선**
데이터: MSR-VTT (9k 학습 / 1k-A 테스트) · 환경: Google Colab A100

- 처음이라면 → [`docs/OVERNIGHT_REPORT.md`](docs/OVERNIGHT_REPORT.md) (지금까지 한 일과 다음 할 일)
- 일정과 결정 → [`docs/ROADMAP.md`](docs/ROADMAP.md) · 현재 상태 → [`docs/HANDOFF.md`](docs/HANDOFF.md)
- 협업 규칙(사용자·Claude·Codex) → [`docs/COLLAB.md`](docs/COLLAB.md)

## 빠른 시작 (Colab)
1. 런타임: GPU A100 + High-RAM
2. `notebooks/00_go_no_go.ipynb` 열기 → 모두 실행. 데이터 다운로드, 프레임 캐시, 공식 체크포인트 재평가, 속도 측정까지 진행합니다.
3. GO가 나오면 `notebooks/01_train_baseline.ipynb`를 seed 24/0/1로 실행합니다(끊기면 재실행하면 이어서 학습).
4. 분석은 `notebooks/02_analysis.ipynb`에서 합니다.

## 로컬 (CPU, 테스트용)
```bash
bash scripts/setup_xpool.sh              # 공식 X-Pool 코드를 고정 커밋으로 clone (수정하지 않음)
pip install -r requirements-colab.txt    # + torch/torchvision/opencv
PYTHONPATH=src python -m pytest -q tests/   # GPU/인터넷 없이 tiny CLIP으로 end-to-end 점검
```

## 구조
```
src/tvr/           재현 파이프라인 (공식 코드를 감싸는 래퍼)
  config.py        공식 인자(기본값 동일) + 프로젝트 전용 인자
  compat.py        2026 환경 호환성 수정 (antialias, position_ids, torch.load)
  frame_cache.py   공식 디코딩과 비트 동일한 프레임 캐시 (build/verify/stats)
  data.py          MSR-VTT dataset/loader (캐시 사용, 테스트 시 frame-order probe)
  trainer_ext.py   공식 Trainer + 로그·경량 체크포인트·epoch 재개
  train.py         학습 (벤치마크 모드 --max_steps)
  evaluate.py      공식 프로토콜 평가 + 분석용 dump
  runlog.py        GPU/메모리/시간 기록 (과제 Dataset Criterion 대응)
analysis/          한계 분석 (analyze_dump.py, failure_gallery.py)
proxy/             고정 특징 기반 아이디어 빠른 선별 (extract_features.py, proxy_train.py)
scripts/           setup_xpool.sh, prepare_msrvtt.py, download_checkpoints.sh, dataset_stats.py, make_notebooks.py
notebooks/         00 go/no-go · 01 baseline 학습 · 02 분석 (make_notebooks.py로 생성)
tests/             프레임 캐시 동일성, 호환성, CPU end-to-end
docs/              ROADMAP, HANDOFF, COLLAB, BASELINE_XPOOL, DATASET_FACTSHEET(+_TABLES), RELATED_WORK, IDEAS
experiments/       LOG.md(실험 기록), results/(작은 JSON 결과)
third_party/xpool  (gitignore) 공식 코드, scripts/setup_xpool.sh가 생성
```

## 인용
- X-Pool: Gorti et al., CVPR 2022, https://github.com/layer6ai-labs/xpool (공식 코드는 수정 없이 사용)
- MSR-VTT: Xu et al., CVPR 2016
