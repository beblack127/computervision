# HANDOFF: 현재 상태와 다음 할 일

> 모든 참여자(사용자/Claude/Codex)가 세션을 시작할 때 읽고, 끝날 때 맨 아래 "세션 로그"에 한 줄을 추가합니다(추가만 함).

## 현재 상태 (2026-11-12 기준)

- 단계: **Phase 0 (go/no-go 대기)**
- 준비 완료: 재현 파이프라인(`src/tvr`), 프레임 캐시, Colab 노트북 00/01/02, 분석/프록시 도구, 데이터 통계, 문서
- 검증 완료(CPU): 테스트 8개 통과. 프레임 캐시가 공식 디코딩과 비트 단위로 같음(합성 영상), 학습·평가·재개·분석·프록시 end-to-end 동작(tiny CLIP)
- **미검증(GPU/실데이터 필요)**: 공식 ckpt 46.9 재현, 실제 학습 속도, MSR-VTT 다운로드 경로 가용성

## 다음 할 일

| # | 담당 | 할 일 | 상태 |
|---|---|---|---|
| 1 | 사용자 | PR `setup/xpool-baseline` 검토 후 merge | ⏳ |
| 2 | 사용자 | Colab A100(High-RAM)에서 `notebooks/00_go_no_go.ipynb` 실행 → `experiments/results/go_no_go.json` 커밋, 아래 "go/no-go 결과"에 요약 붙이기 | ⏳ |
| 3 | Claude | go/no-go 해석 → GO 확정 또는 fallback 결정, ROADMAP §3 연산 예산 갱신 | 2번 이후 |
| 4 | 사용자 | GO이면 `01_train_baseline.ipynb` seed 24 학습 시작 (Proposal 쓰는 동안 돌려두기) | 3번 이후 |
| 5 | 사용자 | X-Pool 논문 정독 (`BASELINE_XPOOL.md`를 옆에 두고) → 모르는 부분은 Claude에게 질문 | ⏳ |
| 6 | 사용자 | MSR-VTT 라이선스/이용 조건을 공식 페이지에서 확인 (`DATASET_FACTSHEET.md`) | ⏳ |
| 7 | Codex | (선택) 노트북 00 실행 중 문제가 생기면 디버깅 | 대기 |

## go/no-go 결과
(노트북 00의 마지막 셀 출력을 여기에 붙여넣기)

## 열린 질문
- Q1. Google Drive 여유 용량? (MSR-VTT zip ~6.5GB + 체크포인트 ~0.6GB × run 수 + dump)
- Q2. Colab compute unit 잔량과 요금제? (full run 10~13회 계획)

## 세션 로그
- 2026-11-12 새벽 · Claude · 초기 셋업: 공식 X-Pool 코드 분석, 호환성 수정 3건 발견·적용(antialias, position_ids, torch.load), 프레임 캐시+동일성 테스트, 학습/평가/재개 래퍼, 분석·프록시 도구, 데이터 통계, 문서 일체, 노트북 3개. 남은 일: GPU에서 go/no-go. 다음 담당: 사용자.
