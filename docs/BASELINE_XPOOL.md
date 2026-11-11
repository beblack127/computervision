# 베이스라인 노트: X-Pool (CVPR 2022)

> 목적: 공식 코드를 직접 읽고 확인한 **사실**만 정리한 문서입니다. 논문 이해와 재현 설정 확인용입니다.
> (보고서 문장으로 그대로 옮기지 말 것. 보고서는 본인 말로 작성.)

- 논문: Gorti et al., *X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval*, CVPR 2022 (arXiv 2203.15086)
- 공식 코드: https://github.com/layer6ai-labs/xpool, 고정 커밋 `bc88718` (2023-04-25). `scripts/setup_xpool.sh`가 `third_party/xpool/`에 clone하며, **원본 코드는 수정하지 않습니다.**
- 공식 체크포인트: README의 Google Drive 링크 (`scripts/download_checkpoints.sh`)

## 1. 방법 요약 (코드 기준)

| 구성 | 코드 위치 | 내용 |
|---|---|---|
| 인코더 | `model/clip_transformer.py` | HuggingFace `openai/clip-vit-base-patch32`. 텍스트 → `get_text_features` (512-d, EOS 토큰), 각 프레임 → `get_image_features` (512-d). **CLIP 전체를 fine-tuning** |
| 풀링 | `modules/transformer.py` | 텍스트 임베딩을 query, 12개 프레임 임베딩을 key/value로 쓰는 **1-head cross-attention**(scaled dot-product, `/sqrt(512)`). 뒤에 LayerNorm → Linear(잔차, dropout) → LayerNorm. 모든 Linear는 **항등행렬로 초기화**, bias 0 |
| 유사도 | `modules/metrics.py` | 텍스트마다 다른 "텍스트-조건부 영상 임베딩"과 텍스트 임베딩의 cosine |
| 손실 | `modules/loss.py` | 대칭 InfoNCE(t2v + v2t 평균). `logit_scale`은 CLIP 값으로 시작하고 학습 중 `log(100)`으로 clamp |
| 최적화 | `train.py` | AdamW(wd 0.2). CLIP 파라미터 lr 1e-6, 새 파라미터 lr 3e-5(9k). cosine 스케줄 + warmup 10%. grad clip 1.0 |

## 2. MSR-VTT-9k 공식 설정 (README 명령어 + `config/all_config.py` 기본값)

| 항목 | 값 |
|---|---|
| 프레임 | 12개, **uniform** 샘플링(구간 중앙 프레임). **학습과 테스트 모두 동일** |
| 해상도/증강 | 학습: RandomResizedCrop(224, scale 0.5–1) + 좌우 반전. 테스트: Resize(224, bicubic) + CenterCrop |
| batch / epoch | 32 / 5 → epoch당 5,625 step, 총 28,125 step (`docs/DATASET_FACTSHEET_TABLES.md`) |
| dropout | 0.3 (pooling transformer) |
| seed | 24 (기본값), `cudnn.deterministic=True` |
| 평가 | **epoch당 10번** 1k-A 테스트셋으로 평가(총 50회) |
| 모델 선택 | 최근 5회 평가의 R@1 이동평균(`R1-window`)이 최고일 때 `model_best.pth` 저장 |
| 보고 수치 | `test.py`로 `model_best.pth` 재평가 → t2v R@1 **46.9** / R@5 72.8 / R@10 82.2 / MdR 2 / MnR 14.3 |

## 3. ⚠️ 보고서에서 짚을 만한 점: 테스트셋으로 모델을 고른다

이 프로토콜에는 validation split이 없습니다. 공식 코드는 **1k-A 테스트셋 점수로 best checkpoint를 고릅니다** (`trainer/trainer.py`, `R1-window`). 그래서 보고된 46.9에는 테스트셋 기반 선택이 섞여 있어, 낙관적으로 편향됐을 수 있습니다.

우리 대응 방침(ROADMAP에 반영됨):
1. 논문과 비교할 때는 **공식 프로토콜**(best-on-test)을 그대로 씁니다. 공정 비교를 위해 우리 방법도 똑같이 적용합니다.
2. 동시에 **마지막 체크포인트**(`model_last.pth`) 성능을 함께 보고합니다. 테스트 정보를 쓰지 않는 수치이고, 우리 방법도 같은 방식으로 보고합니다.
3. 두 수치의 차이 자체가 "베이스라인 평가 방식의 한계"로 분석 가치가 있습니다. 교수님이 말한 genuine understanding을 보여줄 수 있는 지점입니다.

## 4. 2026년 Colab 환경에서 필요한 호환성 수정 (`src/tvr/compat.py`)

방법은 바꾸지 않고, 결과가 원본과 같아지도록 맞추는 수정만 했습니다.

| # | 문제 | 수정 | 근거/검증 |
|---|---|---|---|
| 1 | torchvision ≥0.17은 텐서 resize에서 **antialias=True가 기본**입니다. 원 코드(torchvision 0.9)는 antialias가 없었습니다. 그대로 실행하면 모든 테스트 이미지가 달라집니다(최대 픽셀 차이 ≈0.13). | `antialias=False` 고정 | `tests/test_compat.py` |
| 2 | transformers 4.6.1 체크포인트에는 `*.embeddings.position_ids` 버퍼 2개가 있지만 최신 버전에는 없습니다. 그래서 strict 로딩이 실패합니다. | 해당 2개 key만 제거하고 나머지는 strict 로딩 | `tests/test_pipeline_cpu.py` |
| 3 | PyTorch ≥2.6은 `torch.load`의 기본값이 `weights_only=True`입니다. | 먼저 시도하고 실패하면 fallback | 동일 |
| 4 | 공식 코드는 annotation을 상대경로(`data/MSRVTT/...`)로 엽니다. | 절대경로를 쓰는 dataset subclass | `src/tvr/data.py` |
| 5 | transformers 5.x는 API가 바뀌었습니다. | `transformers==4.46.3` 고정(torch 2.15 CPU에서 테스트 통과) | `requirements-colab.txt` |
| 6 | 원 코드는 CLIP 토크나이저 생성 시 `TOKENIZERS_PARALLELISM=False`를 kwarg로 넘깁니다. | 환경변수로 동일하게 설정 | `src/tvr/compat.py` |

**남은 미세 차이 가능성**(재현 수치가 0.x 정도 어긋날 때 원인 후보):
- OpenCV/FFmpeg 버전 차이에 따른 프레임 seek 결과
- CLIP 토크나이저의 ftfy 정제 여부
- GPU/cuDNN 비결정성

## 5. 속도 최적화: 프레임 캐시 (`src/tvr/frame_cache.py`)

샘플링이 학습과 테스트 모두 `uniform`이라, epoch마다 **같은 12프레임**을 읽습니다. 그래서 한 번만 디코딩해서 uint8로 저장해도 결과가 바뀌지 않습니다. 원본 `VideoCapture.load_frames_from_video`와 **비트 단위로 같다는 것**을 다음으로 확인합니다.
- 합성 영상: `tests/test_frame_cache.py`
- 실제 영상 50개: `python -m tvr.frame_cache verify`

랜덤 증강(crop/flip)은 기존처럼 매 step마다 적용됩니다.

## 6. 저장 방식 변경 (결과 영향 없음)

| 항목 | 공식 | 우리 |
|---|---|---|
| epoch별 체크포인트 | 모델+optimizer 전체(~1.8GB) × 5 | 없음. 대신 `resume_state.pth` 1개를 덮어쓰기(Colab 끊김 대비 재개용) |
| best 체크포인트 | 모델+optimizer | 모델만 (`--save_optimizer`로 원복 가능) |
| 마지막 체크포인트 | 없음 | `model_last.pth` 추가 (위 3번의 "마지막 체크포인트" 보고용) |
| 로그 | stdout/tensorboard | 추가로 `run_meta.json`(GPU, 메모리, 시간, 버전, 커밋)과 `eval_history.jsonl`(50회 평가 전부) |
