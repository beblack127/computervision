# HANDOFF: 현재 상태와 다음 할 일

> 모든 참여자(사용자/Claude/Codex)가 세션을 시작할 때 읽고, 끝날 때 맨 아래 "세션 로그"에 한 줄을 추가합니다(추가만 함).

## 현재 상태 (2026-11-12 기준)

- 단계: **Phase 1 (Proposal) + Phase 2 (재현) 진행 중**. G0는 **GO 확정**(11/12, 사용자). 9k 전체, 축소 없음
- 진행 중: `01_train_baseline.ipynb` seed 24 학습(E001, 11/12 시작, 약 6.1h 예상)
- 검증 완료(GPU/실데이터): 공식 9k ckpt 재평가 t2v R@1 46.9(논문과 동일), MSR-VTT 영상 10,000개 누락 없음, A100에서 0.704 s/step
- 검증 완료(CPU): 테스트 9개 통과(HF `datasets` 설치 환경 포함)

## 다음 할 일

| # | 담당 | 할 일 | 상태 |
|---|---|---|---|
| 1 | 사용자 | PR `setup/xpool-baseline` 검토 후 merge | ✅ (11/12) |
| 2 | 사용자 | Colab A100(High-RAM)에서 `notebooks/00_go_no_go.ipynb` 실행 → `experiments/results/go_no_go.json` 커밋, 아래 "go/no-go 결과"에 요약 붙이기 | ✅ (11/12) |
| 3 | Claude | go/no-go 해석 → GO 확정 또는 fallback 결정, ROADMAP §3 연산 예산 갱신 | ✅ (11/12, GO) |
| 4 | 사용자 | GO이면 `01_train_baseline.ipynb` seed 24 학습 시작 (Proposal 쓰는 동안 돌려두기) | ⏳ 진행 중 (11/12) |
| 4a | 사용자 | seed 24가 끝나면 결과 파일(`experiments/results/xpool_9k_s24/`) 커밋, LOG.md E001 채우기 → seed 0, 1 순서로 학습 | 4번 이후 |
| 4b | Claude | E001 학습 곡선(`eval_history.jsonl`)과 best/last 결과 해석, 재현 표 초안 | 4a 이후 |
| 5 | 사용자 | X-Pool 논문 정독 (`BASELINE_XPOOL.md`를 옆에 두고) → 모르는 부분은 Claude에게 질문 | ⏳ |
| 6 | 사용자 | MSR-VTT 라이선스/이용 조건을 공식 페이지에서 확인 (`DATASET_FACTSHEET.md`) | ⏳ |
| 7 | Codex | (선택) 노트북 00 실행 중 문제가 생기면 디버깅 | 대기 |

## go/no-go 결과
원본: `experiments/results/go_no_go.json` (11/12, 커밋 b90b29c 코드로 실행)

| 항목 | 값 |
|---|---|
| 환경 | NVIDIA A100-SXM4-40GB (39.49GB), CPU 12, RAM 83.5GB · torch 2.14.1+cu130, torchvision 0.29.1, transformers 4.46.3, opencv 4.14.0, Python 3.13.16 |
| 데이터 | drive-cache, 영상 10,000개, 누락 0 · 프레임 캐시 760.1s, 27.6GB |
| 공식 9k ckpt 재평가 t2v | R@1 46.9 · R@5 72.8 · R@10 82.2 · MdR 2.0 · MnR 14.293 (논문: 46.9 / 72.8 / 82.2 / 2 / 14.3) |
| 공식 9k ckpt 재평가 v2t | R@1 44.4 · R@5 73.3 · R@10 84.0 · MdR 2.0 · MnR 8.961 |
| 속도 | 0.704 s/step (189 step 측정), data_wait 0.02%, 평가 1회 44.0s, peak GPU 17.05GB |
| full run 추정 | 6.1h = 학습 28,125 step × 0.704s (5.5h) + 평가 50회 × 44s (0.6h) |
| 자동 판정 | NO-GO (조건 ① R@1 ±1.0 통과, 조건 ② ≤6h를 0.1h 초과) |

## 열린 질문
- Q1. Google Drive 여유 용량? 현재 약 10.2GB 사용(zip 6.55 + 공식 ckpt 9k·7k 각 1.83) + run 1회당 약 3.2GB(끝난 뒤 resume_state 지우면 1.4GB). 무료 15GB면 seed 0 학습 전에 정리 또는 용량 추가 필요 (ROADMAP §3)
- Q2. Colab compute unit 잔량과 요금제? 전체 계획에 A100 약 61–79시간 필요 (ROADMAP §3)

## 세션 로그
- 2026-11-12 새벽 · Claude · 초기 셋업: 공식 X-Pool 코드 분석, 호환성 수정 3건 발견·적용(antialias, position_ids, torch.load), 프레임 캐시+동일성 테스트, 학습/평가/재개 래퍼, 분석·프록시 도구, 데이터 통계, 문서 일체, 노트북 3개. 남은 일: GPU에서 go/no-go. 다음 담당: 사용자.
- 2026-11-12 · Claude · Drive 폴더를 `cv_hw2` → `내 드라이브/컴비 프로젝트`로 변경. 이름에 공백이 있어 노트북은 `/content/cv_drive` 심볼릭 링크(DRIVE_ROOT)로 접근함. 노트북 재생성, 스크립트·문서 경로 갱신. 남은 일: go/no-go 실행. 다음 담당: 사용자.
- 2026-11-12 · Claude · go/no-go 1차 실행 실패 수정: ① 공식 X-Pool 폴더(`datasets/` 등)는 `__init__.py`가 없는 namespace 패키지라 Colab의 HF `datasets`에 가려짐 → `xpool_bridge`가 공식 패키지를 직접 바인딩, 회귀 테스트 추가(HF datasets 설치 환경에서 9/9 통과) ② 캐시 에러 4건은 MSRVTT.zip 안의 macOS `._videoN.mp4`(4KB, 영상 아님) → 추출·캐시에서 제외. 실제 영상 10,000개는 정상 ③ 공식 ckpt는 개당 1.83GB(옵티마이저 포함) → 7k는 `--with-7k`일 때만 다운로드. 1차 실행 관측: 프레임 캐시 793s, 27.6GB, 해상도 전부 240x320. 다음 담당: 사용자(노트북 00 재실행).
- 2026-11-12 · Claude · go/no-go 결과 해석: 공식 ckpt 재평가 t2v 46.9/72.8/82.2/2/14.3으로 논문과 일치, full run 추정 6.1h(자동 판정 NO-GO, 6h 기준 0.1h 초과). 사용자가 9k 전체로 GO 확정. go_no_go.json을 experiments/results/로 이동, LOG B001·B002 기록, ROADMAP §0·Phase 0·§3(A100 61–79h, Drive 용량) 갱신. 남은 일: seed 24 학습 완료 후 결과 커밋, Drive 용량(Q1)·compute unit(Q2) 답변. 다음 담당: 사용자.
