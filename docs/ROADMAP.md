# 로드맵: X-Pool 재현 → 한계 분석 → 개선

마감 (KST 23:59, 지각 시 점수 절반): **Proposal 11/15(일) · Intermediate 11/29(일) · Final 12/18(금)**
기준일: 2026-11-12(목), **11/28 갱신(Final 계획 확정)**. 배점: Proposal 5 · Intermediate 5 · Final 10.

## 0. 확정된 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 주제 | Text-to-Video Retrieval | 영상 주제 선호. 교수님 연구실(Cap4Bridge)과 접점 |
| 베이스라인 | X-Pool (CVPR 2022), 공식 코드 고정 커밋 | 예시 목록에 있음. A100 1장으로 전체 재현 가능 |
| 데이터 | MSR-VTT, **9k 학습 / 1k-A 테스트, 축소 없음** | X-Pool 주 설정. **11/12 GO 확정**(사용자): 공식 ckpt 재평가가 논문 값과 일치, full run 추정 6.1h(기준 6h를 0.1h 초과했으나 허용) |
| 보고 방식 | t2v R@1/5/10, MdR, MnR (+v2t). **best-on-test와 last 둘 다**, seed 3개 평균±표준편차 | `BASELINE_XPOOL.md` §3 |
| 플랜 B | 시간 부족 → 7k 학습(논문 보고값 43.9) / 재현 실패 → 디버그 체크리스트 / 치명적 실패(11/14 전) → QD-DETR | 아래 §4 |
| 재현 결과 (11/28) | 3 seed t2v R@1 best **46.17 ± 0.76**, last 45.80 ± 1.01 → **G1 통과** (논문 46.9와 −0.73) | `experiments/results/reproduction_3seed.md` |
| Intermediate 실험 (11/28) | 한계 = 질의 의존 풀링의 비용과 불균형한 이득. U-Pool(본인) + 2단계 검색·적응형 깊이·QB-Norm·융합(기존). 학습 없는 후처리로는 R@1 개선 없음 | `experiments/results/pooling_budget/`, `IDEAS.md` §2-1 |
| **Final 방법 (11/28, 사용자 승인)** | **주: F1 질의별 혼합 풀링 / 예비: F2 질의 적응형 attention temperature**. T-MASS는 leakage 문제로 보류 | `IDEAS.md` §2-2 |
| Final 연산 예산 | **A100 약 35시간 사용 가능**(사용자 확인) → full run 5회 + proxy·분석 | §3 |

## 1. 일정

```
11/12 목 ─ G0 go/no-go ✅ GO
11/13–15 ─ seed 24 학습 ✅ · Proposal 제출 ✅
─────── 11/23 재조정 (Proposal 피드백 반영, §5) ───────
11/23–25 ─ 02_analysis(seed 24) ✅ · 비용 분석 ✅ · seed 0 학습 ✅
11/25–28 ─ seed 1 학습 ✅ · 3 seed 재현 표(G1 통과) ✅ · 03_pooling_budget(Intermediate 예비 결과) ✅
11/28–29 ─ Intermediate 작성(사용자) · Claude 점검 · 제출 ★ 11/29 23:59
─────── 11/28 Final 계획 확정 (§5) ───────
11/30–12/1 ─ ① 사전 분석(학습 없음, Colab 15분): 질의별 최적 α 상한, 최적 α와 질의 길이 관계, U-Pool/X-Pool 질의별 선택 상한
             ② F1·F2를 src/tvr/에 플래그로 구현 + 테스트(플래그 off = 베이스라인 비트 동일)
             ③ 문헌 재확인: TCMA, EERCF, T-MASS leakage 내용
12/1–3   ─ proxy 선별(CLIP 특징 고정, 풀링만 학습, seed 3개): baseline head vs F1 vs F2, 설정 2~3개씩
12/3 수  ─ G2: 주 방법 1개 + 예비 1개 확정
12/3–6   ─ 주 방법 seed 24 full run (예산이 되면 예비도 seed 24)
12/6 토  ─ G3a: seed 24에서 best·last 모두 베이스라인 평균(46.17 / 45.80) 대비 약 +0.8 이상? 아니면 예비로 전환
12/6–10  ─ 선택 방법 seed 0·1 full run → 3 seed 비교
12/10 수 ─ G3: 3 seed 평균이 best·last 모두 베이스라인 평균 초과, 차이가 표준편차보다 큼
12/10–12 ─ ablation 1개(예: F1에서 혼합 가중치 고정 / U-Pool 보조 손실 제거), 정성 예시, 2단계 검색 비용 재측정
12/13–16 ─ 결과 정리, 그림, Final 작성(사용자)
12/17–18 ─ Claude 점검, 코드 zip, 제출 ★ 12/18 23:59
```

## 2. 단계별 산출물과 담당

### Phase 0. Go/No-Go (11/12)
- [사용자] `notebooks/00_go_no_go.ipynb` 실행 (A100, High-RAM)
- [사용자] `experiments/results/go_no_go.json` 커밋 + `HANDOFF.md`에 요약 붙이기
- [Claude] 결과 해석 → GO/NO-GO 확정, Phase 1 계획 조정
- **G0 통과 조건**: 공식 ckpt 재평가 R@1이 46.9±1.0 이내 **그리고** full run 추정 ≤ 6시간
- **G0 결과 (11/12)**: R@1 46.9(논문과 동일), full run 추정 6.1h → 자동 판정 NO-GO였으나 **GO 확정**. 근거: data_wait 0.02%로 GPU 연산이 병목이라 설정으로 줄일 여지가 없고, 평가 50회는 공식 모델 선택 규칙이라 줄이면 안 됨. 7k로 줄여도 4.9h로 1.2h만 절약되는데 데이터 기준(충실한 재현)에서 손해. 상세: `HANDOFF.md` "go/no-go 결과"

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
- **G1 (11/20 → 11/27로 연기)**: 3 seed 평균 R@1이 46.9±1.0 이내, 또는 차이를 설명 가능(§4 체크리스트). seed 0·1이 늦어지면 Intermediate에는 완료된 seed까지 보고하고 나머지는 Final에서 갱신
- **결과 (11/28) ✅**: 46.17 ± 0.76 → 통과. `reproduction_3seed.md`

### Phase 3. 한계 분석 (11/23–11/24, 피드백 반영으로 확장)
- [사용자] `notebooks/02_analysis.ipynb` 실행 (seed 24 dump로 바로 가능, A100 런타임 권장)
- [Claude] `analysis_report.md`, `cost_report.md` 해석 → `IDEAS.md` 가설 H1~H7 판정, 그림 선정
- [Codex] 추가 분석 스크립트(필요 시), 그림 다듬기
- **3-A 실패 분석** (피드백 ①): 실패 사례 갤러리(최악 + 아슬아슬한 실패), 카테고리/질의 길이/시간어별 성능, frame-order probe, hubness, 라벨 모호성 비율
- **3-B 질의 의존 풀링 vs 비용** (피드백 ②, 새 작업, `analysis/cost_tradeoff.py`)
  - 같은 학습 모델로 풀링만 바꿔 비교: X-Pool(질의 의존) vs 평균(질의 무관) → 질의 의존성이 주는 정확도 이득
  - 질의별 득실: 프레임 다양성 사분위, 질의 길이, 시간어, 카테고리별 이득 → "언제 질의 의존성이 필요한가"
  - 2단계 검색(평균으로 top-P → X-Pool 재정렬, X-Pool 논문 부록 C): P에 따른 R@1과 X-Pool 계산 비율
  - 순위 계산 비용: 갤러리 1k~10k에서 시간·메모리·저장 용량 (평균 vs X-Pool vs 2단계)
- **3-C 연결**: X-Pool이 실패한 질의가 평균 풀링에서는 어땠는가 → 실패의 원인이 "풀링"인지 "인코더 표현"인지 구분
- 산출물: 위 표·그림 + 정성 예시 3~5개 (Intermediate 3점 항목의 근거)
- **결과 ✅**: `experiments/results/xpool_9k_s24/analysis/`(실패 분석, 비용), H1 구조적 확인·지표 영향 없음, H7 확인

### Phase 4. 아이디어 선정과 예비 결과 (11/25–11/27)
- **G2' (11/25)**: Phase 3 결과로 방향 1개 확정. 후보 (`IDEAS.md` §2)
  - D1 시간 순서 인식 (근거: E001 probe)
  - D6 비용을 고려한 질의 의존 풀링 (근거: 3-B, 피드백 ②)
  - 결합안: 시간 정보를 넣되 평균 풀링 수준의 비용 유지 등
  - 안전 트랙 S(T-MASS 방식, 출처 명시)는 그대로 유지
- [Claude] 선정 방향의 설계(수식, 위치, 플래그), 겹침 점검(Prompt Switch·TeachCLIP·T-MASS·Cap4Bridge)
- [Codex] `proxy/proxy_train.py` HEADS에 구현 + 테스트 → 프록시 실행 스크립트
- [사용자] 프록시 실행(분 단위, 우선) → 시간이 되면 full run 1회(6.1h)
- 예비 결과 기준: 과제는 "방향만 보여주면 충분" → 프록시에서 seed 3개 일관된 이득 + 비용 수치
- Intermediate 요구사항: 재현 결과(정량+정성), 실패 분석, 아이디어 방향과 예비 결과, 출처 명시(본인 아이디어인지 차용인지)
- **결과 (11/28) ✅**: G2'는 "비용 한계 + U-Pool"로 결정(D1 시간 순서는 MSR-VTT 지표 영향 근거가 약해 제외). 노트북 03으로 seed 3개 예비 결과 확보: 2단계 P=20이 X-Pool과 같은 R@1, 10k에서 약 61배 빠름 / 적응형 깊이·QB-Norm·융합은 R@1 개선 없음. 문헌 확인으로 적응형 깊이는 "기존"으로 정정

### Phase 5. Final 본 방법 (11/30–12/12) — 11/28 확정
**방법** (`IDEAS.md` §2-2)
- 주 F1 질의별 혼합 풀링: z = g(t)·z_xpool + (1−g(t))·z_upool (같은 풀링 모듈 공유), 손실 = InfoNCE(z) + λ·InfoNCE(z_upool)
- 예비 F2 질의 적응형 attention temperature: logit × α(t), α 초기 1
- 공통: `src/tvr/`에 플래그로 추가, `third_party/xpool` 수정 금지, 새 층은 베이스라인과 같은 출력에서 시작하도록 초기화(F1: g를 1에 가깝게, F2: α=1)

**단계와 담당**
- [Claude] 사전 분석 스크립트(질의별 최적 α 상한, 선택 상한), F1·F2 설계 확정(수식, 초기화, λ 후보), 문헌 재확인, proxy 결과 해석
- [Claude/Codex] 구현 + 테스트: 플래그 off 비트 동일, F1에서 g=1이면 X-Pool과 동일, F2에서 α=1이면 X-Pool과 동일 / proxy HEADS 추가 / 노트북(학습: 01에 플래그 인자, 분석: 03 재사용)
- [사용자] 사전 분석(15분) → proxy(분 단위) → full run(약 6.2h/회) 실행
- 판단 지점: **G2 (12/3)** proxy에서 seed 3개 모두 baseline head 대비 +0.5 이상 → 주·예비 확정 / **G3a (12/6)** seed 24 full run이 best·last 모두 베이스라인 평균 대비 약 +0.8 → 아니면 예비로 전환 / **G3 (12/10)** 3 seed 평균이 best·last 모두 베이스라인 평균 초과, 차이가 표준편차보다 큼
- 실패해도 기록: 모든 run을 `experiments/LOG.md`에 남김(Final은 "안 된 시도" 포함 요구)

### Phase 6. 최종 (12/12–12/18)
- [Claude] 결과 해석, 통계(seed 분산), 2단계 검색 비용 재측정(F1의 U-Pool 경로를 1단계로), 정성 예시 선정, 남은 한계·향후 방향 메모(문장 X), Final 작성 가이드(`docs/FINAL_OUTLINE.md`)
- [Codex] 코드 zip 정리(README 실행법, third_party 포함 여부 결정), 그림 최종화
- [사용자] Final 작성: 시도했으나 실패한 아이디어 포함(Intermediate의 QB-Norm·융합·적응형 깊이 + Final 탈락 후보), 최고 방법과 이유, 시각화, 남은 한계
- Final 요구사항 대응: 베이스라인 초과(같은 분할·지표) 필수, 본인 방법 가산점, 정량+정성, 근거 기반 분석, 남은 한계와 향후 방향

## 3. 연산 예산 (11/28 갱신)

T = **6.1h/run** (실측: seed 0 6.13h, seed 1 6.11h, A100-40GB). F1·F2는 풀링 모듈만 조금 늘어 **약 6.2h로 가정**(첫 run에서 실측으로 교체).

**사용 완료**: 베이스라인 3회 ≈ 19.5h(seed 24 ≥7.27h 포함) + 분석 노트북 02·03 약 1h

**Final 예산: A100 약 35시간 (사용자 확인, 11/28)**
| 항목 | 횟수 | A100 시간 | 비고 |
|---|---|---|---|
| 사전 분석 + 특징 추출 + proxy | – | 약 2h | 특징 추출 1회(학습 9k + 테스트), proxy는 실행당 수 분 |
| 주 방법 seed 24 | 1 | 6.2h | G3a 판단 |
| 예비 방법 seed 24 | 0–1 | 0–6.2h | G3a에서 주 방법이 실패했을 때만 쓰는 것을 기본으로 |
| 선택 방법 seed 0·1 | 2 | 12.4h | 3 seed 비교 |
| ablation | 1 | 6.2h | seed 24 1개 |
| 2단계 비용 재측정·분석 | – | 약 0.5h | |
| **합계** | **4–5** | **약 27–34h** | 여유 1~8h(끊김 재시작 대비) |

Google Drive 용량 (추정, 첫 run 후 실측으로 교체):

| 항목 | 용량 | 비고 |
|---|---|---|
| `data/MSRVTT.zip` | 6.55GB | 고정 |
| `checkpoints/official_9k` | 1.83GB | 고정. `official_7k`(1.83GB)는 플랜 B가 아니면 삭제 가능 |
| run 1회 | 약 3.2GB | model_best·model_last 각 ~0.6GB + resume_state ~1.8GB(모델+옵티마이저) + dump <0.2GB |
| run이 끝난 뒤 | 약 1.4GB | 학습이 끝나면 `resume_state.pth`는 지워도 됨(이어서 학습할 때만 필요) |

주의: resume_state는 저장 중에 임시 파일(.tmp)이 하나 더 생기므로 ~1.8GB 여유가 더 필요. 무료 15GB로는 run 1회 진행 중에도 빠듯함.

## 4. Fallback / 디버그 체크리스트

- **full run이 6시간 초과**: ① num_workers 조정, 캐시 형식 점검(data_wait_fraction 확인) → ② 그래도 길면 7k 학습 설정(논문 보고 43.9, 근거 있는 축소)으로 전환하고 Proposal에 근거를 명시
- **공식 ckpt 재평가가 46.9에서 1.0 이상 벗어남**: ① 프레임 캐시 verify 결과 ② antialias 설정 ③ 토크나이저(ftfy 설치 여부) ④ 영상 개수 누락(`prepare_report.json`) ⑤ position_ids 외 key 불일치 순서로 확인
- **학습 재현이 낮음**(ckpt 재평가는 정상인데 학습 결과만 낮음): seed 분산 확인(3 seed) → 학습 곡선 비교 → lr/스케줄 확인
- **Colab 끊김**: 노트북 01 전체 재실행 → `resume_state.pth`에서 epoch 단위로 이어짐
- **11/14까지 G0 실패 + 원인 불명**: QD-DETR(QVHighlights)로 전환 검토(마지막 수단)
- **Final: proxy에서 F1·F2 모두 이득 없음(12/3)**: ① 초기화 확인(플래그 on이어도 시작점이 베이스라인과 같은지) ② F1의 λ·g 구조 변형 1~2개 ③ 그래도 없으면 full run에서 바로 확인(proxy는 인코더가 고정이라 효과를 과소평가할 수 있음)
- **Final: G3a 실패(12/6)**: 예비 방법으로 전환해 seed 24부터 다시. 예산상 이 경우 ablation을 생략
- **Final: G3 실패(12/10)**: 결과를 정직하게 보고하는 것이 우선. 최후 수단인 seed 앙상블은 계산량이 3배라 공정 비교가 아님을 명시해야 하며 "본 방법"으로 내세우지 않음

## 5. 변경 이력
- **11/23 · Proposal 피드백 반영** (교수님 피드백: 직접 재현 후 ① 검색 실패 분석 ② 질의 의존 풀링과 계산 비용의 트레이드오프 분석이 개선 방향에 도움)
  - Phase 3에 3-B(비용-정확도 분석)와 3-C(실패와 풀링 연결) 추가, 새 스크립트 `analysis/cost_tradeoff.py`, 노트북 02에 셀 추가
  - 개선 방향 후보에 D6(비용 인식) 추가, 최종 선정은 분석 후 11/25 (G2')
  - 일정 재조정: seed 0·1 지연 반영, 예비 결과는 프록시 우선
- **11/28 · Final 계획 확정** (사용자 승인, A100 약 35h 확인)
  - Phase 2~4 완료 반영: G1 통과(46.17 ± 0.76), Intermediate 예비 결과(노트북 03)
  - Final 주 방법 F1(질의별 혼합 풀링), 예비 F2(질의 적응형 temperature). 근거: 학습 없는 후처리 실패, 오라클 52.0, α 곡선
  - 문헌 확인: 질의 적응형 temperature와 질의 무관+조건 결합은 TCMA(arXiv 2510.10180)에 있음 → F2는 기존 적용, F1은 범위를 좁혀 본인 기여 주장. T-MASS 공식 코드는 "data leakage 확인, 사용 금지"로 보관 표시 → 안전 트랙에서 제외
  - 일정: proxy 선별(G2 12/3) → seed 24 full(G3a 12/6) → seed 0·1(G3 12/10) → ablation·분석 → 12/18 제출
