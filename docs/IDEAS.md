# 한계 가설과 아이디어 풀

> 상태: **가설 단계**입니다. (11/23: Proposal 피드백을 반영해 H7·I6 추가, 선정은 Phase 3 분석 후 11/25) 재현 결과가 나오기 전이라 아래 내용은 모두 검증 전이며, 분석 결과를 보고 고르거나 버립니다.
> 원칙: (1) 분석으로 한계를 **수치로 보인 다음** 아이디어를 고른다. (2) 아이디어마다 **겹침 점검**(RELATED_WORK §3, 특히 Cap4Bridge·T-MASS)을 한다. (3) 프록시 선별 → full run 순서로 진행한다.

## 1. 한계 가설 → 확인 방법

| ID | 가설 | 확인 도구 | "한계가 맞다"로 볼 기준 |
|---|---|---|---|
| H1 | **시간 순서를 쓰지 못한다.** X-Pool 풀링은 프레임 순서를 바꿔도 결과가 같은 구조라, 순서가 중요한 질의("…한 뒤 …한다")에 약하다 | `evaluate --frame_order reversed/shuffled/middle`, 시간어 포함 질의 63개 분석(`analyze_dump.py`) | reversed/shuffled R@1이 원본과 사실상 같음(≈구조적 불변) + `middle`(프레임 1장 반복)과의 격차 + 시간어 질의 R@1이 평균보다 낮음 |
| H2 | **문장 전체를 벡터 하나로 줄인다.** 질의 속 세부 단어(물체, 색, 수량)를 놓친다 | 실패 사례 갤러리, 질의 길이별 성능 | 긴 질의/세부 묘사 질의에서 성능 하락, top-1 오답이 "주제는 맞고 세부가 틀린" 경우가 많음 |
| H3 | **hubness.** 소수 영상이 많은 질의에서 반복 검색된다 | `analyze_dump.py` (N_10 skewness, never-top1 개수), 진단용 dual-softmax 개선폭 | 진단용 재정규화로 R@1이 크게 오름 → 학습 단계에서 다룰 여지가 있다는 신호 |
| H4 | **정답 라벨이 모호하다.** 다른 영상의 캡션이 질의와 거의 같아(일대다), 실제로는 "정답 같은 오답"이 많다 | `analyze_dump.py` failure taxonomy (질의와 오답 영상 캡션의 cos ≥ 0.9 비율) | 실패의 상당 비율이 텍스트상 거의 동일 → 대조학습의 거짓 음성(false negative) 문제 |
| H5 | **12프레임 균일 샘플링이 짧은 사건을 놓친다** | 실패 갤러리의 attention 분포, 영상 길이별 성능 | 긴 영상에서 성능 하락, attention이 특정 프레임에 몰렸는데도 실패 |
| H6 | **attention 온도가 고정**이라, 질의 특이성과 상관없이 프레임 attention의 날카로움이 같다 | attention 엔트로피 vs 질의 길이/정답 여부 | 실패 질의에서 attention이 과도하게 평평하거나 과도하게 뾰족함 |
| H7 | **질의 의존 풀링은 비싸고, 그 이득이 모든 질의에 고르게 돌아가지 않는다** (Proposal 피드백 ②) | `analysis/cost_tradeoff.py`: 같은 모델에서 X-Pool vs 평균 풀링 정확도, 질의별 득실(프레임 다양성·질의 길이·시간어), 2단계 검색 곡선, 갤러리 크기별 시간·메모리 | 이득이 특정 질의/영상 유형(예: 장면이 다양한 영상)에 몰려 있고, 작은 P의 2단계 검색으로도 대부분의 이득이 회수되면 → "질의 의존성을 필요한 곳에만" 쓸 여지가 있음

## 2. 아이디어 후보 (가설과 연결)

| ID | 아이디어 | 대응 가설 | 프록시로 선별 가능? | 겹침 점검 메모 |
|---|---|---|---|---|
| I1 | 순서 정보를 반영하는 풀링: 상대 시간 위치 + **프레임 순서 섞은 영상을 hard negative**로 쓰는 손실 | H1 | 부분적으로 가능 (`xpool_tpe` 예시 head 있음; hard negative 손실은 full run 필요) | 시간 순서 민감성을 다루는 영상-언어 연구가 있음 → 문헌 확인 필수 |
| I2 | 질의별 적응형 attention 온도 | H6, H2 | 가능 | X-Pool 원 논문은 고정 scaling; 후속 연구 점검 필요 |
| I3 | 학습 단계의 hubness 억제 정규화 (테스트 후처리 아님) | H3 | 가능 | QB-Norm/DSL은 테스트 시 후처리, UCoFiA는 Sinkhorn 사용 → 차별점 명확히 |
| I4 | 캡션-캡션 유사도로 **거짓 음성 완화**(soft target / 다중 정답) | H4 | 가능 | 대조학습 false negative 연구 다수 → 영상검색 특화 지점 찾기 |
| I5 | (참고) 확률적 텍스트 임베딩 | H2 | 가능 | **T-MASS와 동일 방향**. 쓴다면 "adapted from T-MASS"로 명시하고 안전 트랙에서만 |
| I6 | **비용을 고려한 질의 의존 풀링**: 질의 의존 계산이 필요한 후보/질의에만 X-Pool을 쓰거나(적응형 shortlist), 질의 의존 계산 자체를 가볍게 만들어 정확도-비용을 함께 개선 | H7 | 부분적으로 가능 (2단계/적응형 shortlist는 학습 없이 dump로 평가 가능) | X-Pool 부록 C(고정 P 2단계), Prompt Switch(융합 제거), TeachCLIP(증류)와 차별화 필요 |

표기 대응: 채팅과 ROADMAP의 **D1 = I1**(시간 순서), **D6 = I6**(비용 인식), **S = 안전 트랙**(I5, T-MASS 방식).

### 2-1. Intermediate 선택 (11/28): I6을 문헌 확인 후 재구성 → `analysis/pooling_budget.py`, 노트북 03

문헌 확인 결과(11/28 검색):
- **고정 P 2단계**(평균 풀링 shortlist → 질의 조건 풀링 재정렬)는 X-Pool 부록 C와 Bain et al. 2022("A CLIP-Hitchhiker's Guide to Long Video Retrieval")에 이미 있음 → **기존 연구(E1)**.
- **질의별 재정렬 깊이 조절**(첫 단계 점수의 확신도로 shortlist 길이 결정)은 IR의 ranked-list truncation / dynamic cutoff 계열(Culpepper et al. ADCS 2016, Lien et al. ICTIR 2019, Bahri et al. SIGIR 2020 Choppy, Meng et al. SIGIR 2024)과 adaptive re-ranking·early-exit 연구, 영상 검색의 confidence 기반 cascade(CoVRAGE, arXiv 2609.10008)에 이미 있음 → **기존 아이디어를 가져온 것(E2, adapted)**. 처음에 "우리 아이디어"로 제안했던 것은 철회.
- **U-Pool**(학습된 X-Pool head에 균일 attention을 넣어 질의 무관 벡터로 만들고, 그것으로 X-Pool 이득을 "투영층 몫 vs 질의 조건 attention 몫"으로 분해 + 첫 단계로 사용): 검색 범위(X-Pool 본문·부록, Bain et al. 2022, EERCF, ProCLIP, Prompt Switch, TeachCLIP, QB-Norm, RLT 계열)에서 같은 것을 찾지 못함 → **우리 아이디어(O1/O2)**로 제시하되, "균일 가중치 대조군"은 다른 논문(예: SCALAR, arXiv 2609.15335)에도 흔한 ablation이라 그 점을 밝히고, 주장은 "학습된 X-Pool head를 그대로 질의 무관 인덱스로 재사용"과 "그 결과로 얻는 정확도-비용 곡선"에 한정.

| ID | 아이디어 | 대응 | 출처 구분 | 평가 |
|---|---|---|---|---|
| E1 | 평균 풀링 shortlist(고정 P) → X-Pool 재정렬 | L2 비용 | 기존 (X-Pool 부록 C; Bain et al. 2022) | 학습 없음, seed 3개 |
| E2 | 첫 단계 top1–top2 점수 차로 질의별 P 결정(평균 예산을 라벨 없이 맞춤) | L2 비용 | 기존 아이디어 적용 (RLT/dynamic cutoff) | 학습 없음 |
| I7 = O1/O2 | **U-Pool**: X-Pool head + 균일 attention = 질의 무관·사전 계산 가능한 영상 벡터. ① 이득 분해(mean → U-Pool → X-Pool), ② attention 날카로움(logit × α) 곡선, ③ 첫 단계로 사용(+E2) | L2 비용, L3 상보성, H6 | 우리 아이디어 (위 검색 범위 기준) | 학습 없음 |


## 3. 선별 절차

1. **프록시**: 재현 모델의 CLIP 특징을 고정하고 head만 학습(`proxy/`, 실행당 수 분). 아이디어마다 seed 3개를 돌려 프록시 baseline(`xpool` head)과 비교합니다. 평균 이득이 **+0.5 R@1 이상이고 seed 간 일관**되면 통과입니다.
2. **Full run**: 통과한 1~2개만 end-to-end로 학습합니다(9k, 공식 하이퍼파라미터, seed 3개).
3. **최종 판단**: 공식 프로토콜(best-on-test)과 마지막 체크포인트 **둘 다**에서 베이스라인 평균보다 높아야 합니다.
4. 실패한 아이디어도 `experiments/LOG.md`에 남깁니다. 최종 보고서는 "시도했으나 안 된 것"까지 요구합니다.

## 4. 안전 트랙 vs 도전 트랙

- **도전 트랙(가산점)**: I1~I4 중 분석 근거가 가장 강한 것 1개를 본인 아이디어로 발전시킵니다.
- **안전 트랙(필수 조건 충족용)**: 도전 트랙이 11/30까지 프록시에서 이득을 못 보이면, 후속 연구 기법(예: I5 T-MASS 방식)을 출처를 명시하고 적용해 "베이스라인 초과"를 확보합니다.
