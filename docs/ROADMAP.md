# 로드맵: X-Pool 재현 → 한계 분석 → 개선

마감 (KST 23:59, 지각 시 점수 절반): **Proposal 11/15(일) · Intermediate 11/29(일) · Final 12/18(금)**
기준일: 2026-11-12(목). 배점: Proposal 5 · Intermediate 5 · Final 10.

## 0. 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 주제 | Text-to-Video Retrieval | 영상 주제 선호. 교수님 연구실(Cap4Bridge)과 접점 |
| 베이스라인 | X-Pool (CVPR 2022), 공식 코드 고정 커밋 | 예시 목록에 있음. A100 1장으로 전체 재현 가능 |
| 데이터 | MSR-VTT, **9k 학습 / 1k-A 테스트, 축소 없음** | X-Pool 주 설정. go/no-go로 시간 확인 후 확정 |
| 보고 방식 | t2v R@1/5/10, MdR, MnR (+v2t). **best-on-test와 last 둘 다**, seed 3개 평균±표준편차 | `BASELINE_XPOOL.md` §3 |
| 플랜 B | 시간 부족 → 7k 학습(논문 보고값 43.9) / 재현 실패 → 디버그 체크리스트 / 치명적 실패(11/14 전) → QD-DETR | 아래 §4 |

## 1. 일정

```
11/12 목 ─ G0 go/no-go (Colab, 사용자 1~1.5h) · PR merge
11/13 금 ─ 논문 정독 + Q&A(Claude) · baseline seed24 학습 시작(GO일 때)
11/14 토 ─ Proposal 작성(사용자) · Claude 사실 점검 · seed0 학습
11/15 일 ─ Proposal 제출 ★ · seed1 학습
11/16–20 ─ 재현 완료(3 seed), 공식 ckpt 재평가, probe 평가 → G1
11/19–24 ─ 한계 분석(analysis/, gallery), 프록시 특징 추출, 가설 판정
11/22–27 ─ 프록시로 아이디어 3~5개 선별 → G2 → 최상위 1개 full run(예비 결과)
11/27–29 ─ Intermediate 작성 · 제출 ★
11/30–12/11 ─ 본 방법 확정: full run 3 seed + ablation 3~5개 → G3
12/12–17 ─ 최종 분석(시각화, 정량/정성), Final 작성, 코드 zip 정리
12/18 금 ─ Final 제출 ★ (예비일: 12/16–17)
```

## 2. 단계별 산출물과 담당

### Phase 0. Go/No-Go (11/12)
- [사용자] `notebooks/00_go_no_go.ipynb` 실행 (A100, High-RAM)
- [사용자] `experiments/results/go_no_go.json` 커밋 + `HANDOFF.md`에 요약 붙이기
- [Claude] 결과 해석 → GO/NO-GO 확정, Phase 1 계획 조정
- **G0 통과 조건**: 공식 ckpt 재평가 R@1이 46.9±1.0 이내 **그리고** full run 추정 ≤ 6시간

### Phase 1. Proposal (~11/15)
과제가 요구하는 세 항목과 우리 근거:

| 배점 항목 | 담아야 할 것 | 근거 자료 |
|---|---|---|
| Topic & Dataset [2] | 문제 정의, 흥미로운 이유, MSR-VTT 출처·규모·라이선스·전처리·분할·**축소 없음**·연산 환경 | `DATASET_FACTSHEET.md`, `_TABLES.md`, `go_no_go.json` |
| Related paper review [2] | X-Pool(방법 개요) + CLIP4Clip(배경) + 후속 연구 2~3개(T-MASS, UCoFiA, Cap4Video) | 논문 원문, `RELATED_WORK.md` |
| Plan [1] | 재현 방법(공식 설정, seed 3개, 두 가지 보고 방식), 일정, 개선 방향 후보(가설 수준) | 이 문서, `IDEAS.md` |

- [사용자] 작성 (한국어, Word)
- [Claude] 논문 Q&A, 사실 확인, 논리 점검 코멘트(문장 작성 X)
- [Codex] 필요 시 그림/표 생성 스크립트

### Phase 2. 재현 (11/13–11/20)
- [사용자] `notebooks/01_train_baseline.ipynb` × seed {24, 0, 1}. 끊기면 재실행하면 이어서 학습
- [Claude] `eval_history.jsonl` 학습 곡선 분석, 재현 표(논문 vs 우리 3 seed), 불일치 원인 진단
- [Codex] 학습 중 오류 디버깅, 파이프라인 개선(결과 불변 조건)
- **G1 (11/20)**: 3 seed 평균 R@1이 46.9±1.0 이내, 또는 차이를 설명 가능(§4 체크리스트)

### Phase 3. 한계 분석 (11/19–11/24)
- [사용자] `notebooks/02_analysis.ipynb` 실행
- [Claude] `analysis_report.md` 해석 → `IDEAS.md` 가설 H1~H6 판정, 그림 선정
- [Codex] 추가 분석 스크립트(필요 시), 그림 다듬기
- 산출물: 실패 사례 갤러리, 카테고리/질의 길이/시간어별 성능, frame-order probe, hubness 그림

### Phase 4. 아이디어 선별과 예비 결과 (11/22–11/27)
- [Claude] 가설 판정 → 아이디어 2~4개 설계(수식, 위치, 플래그), 겹침 점검
- [Codex] `proxy/proxy_train.py` HEADS에 구현 + 테스트 → 프록시 실행 스크립트
- [사용자] 프록시 실행(분 단위) → **G2**: +0.5 R@1 이상 & seed 일관 → 최상위 1개 full run 1회(예비 결과)
- Intermediate 요구사항: 재현 결과(정량+정성), 실패 분석, 아이디어 방향과 예비 결과, 출처 명시(본인 아이디어인지 차용인지)

### Phase 5. 본 방법 (11/30–12/11)
- full run 3 seed + ablation(구성요소 on/off, 하이퍼파라미터 1~2개)
- **G3 (12/10)**: best-on-test와 last 모두에서 베이스라인 3 seed 평균 초과. 실패하면 안전 트랙(§`IDEAS.md` 4) 가동
- [Codex] `src/tvr/`에 플래그로 통합, 비트 동일성 테스트(플래그 off = 베이스라인)

### Phase 6. 최종 (12/12–12/18)
- [Claude] 결과 해석, 통계(seed 분산), 남은 한계와 향후 방향 정리용 메모(문장 X)
- [Codex] 코드 zip 정리(README 실행법, third_party 포함 여부 결정), 그림 최종화
- [사용자] Final 작성: 시도했으나 실패한 아이디어 포함, 최고 방법과 이유, 시각화, 남은 한계

## 3. 연산 예산 (G0 후 갱신)

| 항목 | full run 횟수 | 비고 |
|---|---|---|
| 베이스라인 재현 | 3 | seed 24/0/1 |
| 예비 결과 | 1–2 | Intermediate용 |
| 본 방법 | 3 | 동일 seed |
| ablation | 3–5 | 가능하면 seed 1개 |
| **합계** | **10–13 runs × T시간** | T = go_no_go 추정치. 프록시 실험은 별도(무시 가능 수준) |

## 4. Fallback / 디버그 체크리스트

- **full run이 6시간 초과**: ① num_workers 조정, 캐시 형식 점검(data_wait_fraction 확인) → ② 그래도 길면 7k 학습 설정(논문 보고 43.9, 근거 있는 축소)으로 전환하고 Proposal에 근거를 명시
- **공식 ckpt 재평가가 46.9에서 1.0 이상 벗어남**: ① 프레임 캐시 verify 결과 ② antialias 설정 ③ 토크나이저(ftfy 설치 여부) ④ 영상 개수 누락(`prepare_report.json`) ⑤ position_ids 외 key 불일치 순서로 확인
- **학습 재현이 낮음**(ckpt 재평가는 정상인데 학습 결과만 낮음): seed 분산 확인(3 seed) → 학습 곡선 비교 → lr/스케줄 확인
- **Colab 끊김**: 노트북 01 전체 재실행 → `resume_state.pth`에서 epoch 단위로 이어짐
- **11/14까지 G0 실패 + 원인 불명**: QD-DETR(QVHighlights)로 전환 검토(마지막 수단)
