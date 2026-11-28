# Intermediate Report 작성 가이드: 목차 · 개조식 재료 · 작성 팁

> 마감: **2026-11-29(일) 23:59 KST**. 한국어 Word → PDF로 LMS 제출.
> 배점: 총 5점 = Proposal 요약(배점 없음, 필수) + **Baseline reproduction 2점** + **Limitation analysis & initial idea 3점**
> 작성 기준일: 11/28. 수치는 모두 `experiments/results/`의 JSON/MD에서 가져왔고, 출처 파일을 옆에 적었습니다.
> 제출한 Proposal(`JaehyunKim_Assignment2_Proposal_Report.docx`)을 읽고 이어지도록 구성했습니다.

---

## 0. 이 문서 사용법 (먼저 읽기)

**과제 규정** (HW2.pdf)
- AI는 논문 이해와 코딩 보조에만 쓸 수 있고, **보고서는 전부 본인 말로** 써야 합니다.
- 논리 오류, 일관되지 않거나 부적절한 표현, 이해 부족이 드러나면 위반으로 보고 크게 감점합니다.
- 각 보고서는 이전 보고서의 **짧은 업데이트 버전**을 포함해야 합니다.

그래서 이 문서는 Proposal 때와 같이 **완성 문장이 아니라 "쓸 재료"**입니다. bullet 하나가 사실 하나 또는 논점 하나이고, 본인 문장으로 풀어 쓰면 됩니다.

| 표시 | 뜻 |
|---|---|
| `[논문]` | 해당 논문의 수치/주장. 본문에 인용 번호 필요 |
| `[측정]` | 우리가 직접 실행해서 얻은 수치. 옆에 근거 파일 |
| 💭 **[해석]** | **본인 판단을 쓰는 자리**. 재료와 질문만 줍니다 |
| ✅ 확인 | 원문/그림을 직접 열어 보고 확인할 항목 |
| 📊 | 표나 그림 자리 (§2에 목록) |
| ⚠️ | 틀리기 쉬운 지점. 감점 위험 |

**과제 원문의 요구사항** (HW2.pdf, Intermediate)
- Proposal의 업데이트된 짧은 버전 포함
- Baseline reproduction [2pt]: 선택한 벤치마크에서 베이스라인을 돌린 결과(**정량 수치 + 정성 예시**), **재현에 필요한 모든 세부사항**
- Limitation analysis & initial idea [3pt]: 재현 결과를 바탕으로 **어디서, 왜** 실패하거나 부족한지(예: 실패 사례), **한계 하나**를 해결할 접근 제안, **예비 결과**(완성된 해법은 필요 없고 방향이면 충분), **본인 아이디어인지 기존 연구를 가져온 것인지 명시**(둘 다 허용, 출처 인용 필수)
- 데이터셋 기준(공통): GPU 모델과 메모리, batch size, epoch 수, **메모리 사용량, 총 학습 시간** 보고

**교수님 Proposal 피드백** (반영했다는 점이 드러나야 함)
- 방법·결과 비교, 공식 체크포인트로 미리 검증한 점은 좋음
- 베이스라인을 직접 재현한 뒤 **검색 실패 분석**, 그리고 **질의 의존 풀링과 계산 비용 사이의 trade-off**를 분석하면 개선 방향을 잡는 데 도움이 됨

**분량 감각** (자세히 쓰고 나중에 줄이는 전제): 1장(Proposal 요약) 1.5~2쪽, 2장(재현) 3~4쪽, 3장(한계·아이디어) 5~6쪽, 참고문헌 1쪽

---

## 1. 전체 목차

```
표지 (과목, Assignment #2, Intermediate Progress Report, 학과/학번/이름)
0. 요약 (선택, 5~6줄)
1. Proposal 요약 및 변경 사항
   1.1 문제와 동기
   1.2 데이터셋과 분할 (표 1)
   1.3 베이스라인 X-Pool과 관련 연구 (표 2: 비교 요약)
   1.4 Proposal 이후 바뀐 점 (피드백 반영, 정정)
2. 베이스라인 재현 [2pt]
   2.1 재현 설정 (표 3: 하이퍼파라미터·코드·버전)
   2.2 재현 세부사항: 환경 호환 수정, 프레임 캐시, 평가와 모델 선택
   2.3 연산 환경과 실제 사용량 (표 4)
   2.4 정량 결과 (표 5: 3 seed, best/last, 공식 ckpt 비교)
   2.5 학습 곡선 (그림 1)
   2.6 정성 결과 (그림 2~3: 성공에 가까운 예시, 실패 예시)
   2.7 재현 결과 해석
3. 한계 분석과 초기 아이디어 [3pt]
   3.1 분석 방법
   3.2 어디서 실패하나 (표 6, 그림 4)
   3.3 왜 실패하나: 원인별 정리
   3.4 다룰 한계 선정: 질의 의존 풀링의 비용과 불균형한 이득
   3.5 접근 A (본인 아이디어): U-Pool (표 7, 그림 5)
   3.6 접근 B (기존 연구 적용): 2단계 검색, 적응형 깊이, QB-Norm, 점수 융합 (표 8, 그림 6)
   3.7 예비 결론과 Final 계획
   3.8 출처 구분 (표 9: 본인 vs 기존)
참고문헌
AI 도구 사용 고지
```

---

## 2. 섹션별 개조식 재료

### 0. 요약 (선택)
- 담을 것 4가지: 재현 결과 한 줄(3 seed 평균 ± 표준편차, 논문 대비 차이), 주요 실패 원인 2~3개, 다룬 한계 1개, 예비 결과 한 줄(되는 것 + 안 된 것)
- 💭 [해석] 이번 보고서의 핵심 메시지 한 문장을 먼저 정하고, 본문 결론과 일치시킬 것

---

### 1. Proposal 요약 및 변경 사항 (짧게)

> 과제가 "짧은 업데이트 버전"을 요구합니다. Proposal 문장을 그대로 붙이기보다 줄이고, **바뀐 점을 분명히** 하는 쪽이 좋습니다.

#### 1.1 문제와 동기 (3~5줄)
- 과제: 텍스트→영상 검색(t2v), 보조로 v2t. 질의 문장이 영상의 일부 장면만 묘사한다는 점이 핵심 어려움
- 동기: Proposal의 축구 예시(골 장면이 짧게 등장) 한 줄로 축약 가능
- 💭 [해석] 재현·분석을 거친 지금, 동기를 한 줄 보강할 수 있음. 예: "질의에 맞춘 풀링이 실제로 얼마나 기여하는가"라는 질문이 생긴 이유

#### 1.2 데이터셋과 분할 📊 표 1 (Proposal 표 1 재사용 + 수치 보강)
- MSR-VTT (Xu et al., CVPR 2016) `[논문]`: 영상 10,000개, 캡션 200,000개, 카테고리 20개 `[측정: dataset_stats.json]`
- 분할: 학습 9k(영상 9,000개, 문장-영상 쌍 180,000개) / 테스트 1k-A(영상 1,000개, 질의 1,000개), **겹침 0** `[측정: dataset_stats.json]`
- 9k 학습셋 중 1,990개는 2016년 원래 test 분할에서 온 영상 `[측정]` (Proposal에 이미 있음)
- **validation 분할 없음** → 공식 코드는 1k-A 테스트 점수로 best 체크포인트를 고름(§2.2에서 다시 다룸)
- 전처리: 영상당 12프레임 균일 추출, 학습 RandomResizedCrop(224)+좌우 반전 / 평가 Resize(224)+CenterCrop, CLIP 정규화, BPE 토크나이저(최대 77토큰) `[BASELINE_XPOOL.md §2]`
- 데이터 축소/샘플링 없음 (Proposal과 동일)

#### 1.3 베이스라인과 관련 연구 📊 표 2 (Proposal 표 "2.5 종합" 축약)
- X-Pool (Gorti et al., CVPR 2022): 텍스트 임베딩을 query, 12개 프레임 임베딩을 key/value로 쓰는 cross-attention으로 **질의마다 다른 영상 벡터**를 만듦. MSR-VTT 9k t2v R@1 46.9 `[논문]`
- 관련 연구 요약은 Proposal 표를 한 줄씩만: CLIP4Clip 43.1(mean)/44.5(seqTransf), Prompt Switch 46.1(47.8 with X-Pool), T-MASS 50.2 `[논문]`
- **새로 추가할 관련 연구** (3장에서 쓰는 것, 참고문헌에도 추가):
  - Bain et al. 2022 (CLIP-Hitchhiker): 질의 무관 집계로 전체 순위 → 상위 K개만 질의 조건 집계로 재정렬 `[논문]`
  - QB-Norm (Bogolin et al., CVPR 2022): 학습 질의 묶음(querybank)으로 hubness를 줄이는 재정규화, 재학습 불필요 `[논문]`
  - Ranked-list truncation 계열 (Lien et al. ICTIR 2019 등): 질의별로 재정렬 깊이를 정함 `[논문]`

#### 1.4 Proposal 이후 바뀐 점 ⚠️ (반드시 넣기)
- **피드백 반영**: 실패 분석(3.2~3.3) + 질의 의존 풀링 vs 계산 비용 분석(3.4~3.6)을 추가했다는 점을 명시
- **정정 1 – GPU**: Proposal §3.4는 "A100 80GB"로 적었으나 실제로는 seed 24는 A100-SXM4-80GB, seed 0·1은 **A100-SXM4-40GB** `[측정: run_meta.json]`. 최대 사용 메모리가 17.75GB라 40GB로 충분했음
- **정정 2 – CLIP4Clip 해석**: Proposal §2.1.3의 "프레임 사이의 관계를 고려하는 방식이 단순 평균보다 효과적"은 과일반화. 같은 표에서 seqLSTM 42.5, tightTransf 40.2는 meanP 43.1보다 낮음 `[논문: CLIP4Clip 표]` ✅ 원 표 다시 확인. → "seqTransf는 meanP보다 높았지만, 관계를 모델링하는 모든 방식이 더 좋았던 것은 아니다" 정도로 범위를 좁히기
- **정정 3 – 사소한 표기**: §1.1 "묘사될 수 있다.." 마침표 중복
- **보강 – 인용 번호**: Proposal 본문에는 [1], [2] 같은 인용 표시가 거의 없었음. Intermediate에서는 수치·주장마다 번호를 달기
- 💭 [해석] 정정 사항을 "Proposal 이후 변경 사항" 소제목으로 짧게 모으면, 오히려 꼼꼼함이 드러남

---

### 2. 베이스라인 재현 [2pt]

> 만점 조건: ① 정량 수치 ② 정성 예시 ③ **다른 사람이 같은 결과를 낼 수 있을 만큼의 세부사항** ④ 연산 환경과 실제 사용량. 하나라도 빠지면 감점 위험.

#### 2.1 재현 설정 📊 표 3
| 항목 | 값 | 근거 |
|---|---|---|
| 코드 | X-Pool 공식 저장소, 커밋 `bc88718`, 원본 수정 없음 (래퍼 `src/tvr/`에서 호출) | `run_meta.json: xpool_commit` |
| 백본 | CLIP ViT-B/32 (HuggingFace `openai/clip-vit-base-patch32`), **CLIP 전체 미세조정** | `BASELINE_XPOOL.md §1` |
| 풀링 | 1-head cross-attention + LayerNorm + Linear(잔차, dropout 0.3) + LayerNorm, 새 층은 항등 초기화 | 같음 |
| 파라미터 수 | 전체 152,593,665 / 풀링 모듈 1,316,352 | `run_meta.json` |
| 프레임 | 12개 균일 추출 (학습·평가 동일) | `config` |
| batch / epoch / step | 32 / 5 / epoch당 5,625, 총 28,125 | `run_meta.json` |
| 최적화 | AdamW, weight decay 0.2, CLIP lr 1e-6, 새 파라미터 lr 3e-5, warmup 10%(2,812 step) 후 cosine, grad clip 1.0 | `run_meta.json: config`, `BASELINE_XPOOL.md` |
| 손실 | 대칭 InfoNCE (t2v + v2t) | `BASELINE_XPOOL.md` |
| seed | 24(공식 기본값), 0, 1 | `run_meta.json` |
| 평가 | epoch당 10회, 총 50회 1k-A 평가 | `eval_history.jsonl` (seed마다 50행) |
| 소프트웨어 | Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1, transformers 4.46.3, CUDA 13.0 | `run_meta.json: env` |
- ⚠️ 논문 본문과 공식 코드의 차이: 논문은 새 파라미터 lr **1e-5**, warmup 언급 없음 / 공식 README 명령은 **3e-5**, 코드 기본값 warmup 10%. 우리는 공식 명령을 따름(Proposal §3.2에서 이미 밝힘 → 한 줄 재확인)

#### 2.2 재현 세부사항 (재현성의 핵심)
- **2026년 환경 호환 수정** (방법은 바꾸지 않고 원래 결과가 나오게 맞추는 수정만) `[BASELINE_XPOOL.md §4]`:
  - torchvision 최신판의 resize 기본값 antialias=True → 원 코드와 같도록 **False로 고정** (안 하면 모든 테스트 이미지가 달라짐, 최대 픽셀 차이 ≈0.13)
  - 옛 체크포인트의 `position_ids` 버퍼 2개만 제거하고 나머지는 strict 로딩
  - `torch.load` 기본값 변경(weights_only) 대응
  - transformers 4.46.3으로 고정 (5.x는 API 변경)
- **프레임 캐시**: 학습·평가 모두 같은 12프레임을 쓰므로 한 번만 디코딩해 저장. 원본 디코딩과 **비트 단위로 동일**함을 테스트로 확인 `[tests/test_frame_cache.py]`. 랜덤 증강은 매 step 그대로 적용
- **사전 검증**: 공식 체크포인트를 우리 평가 코드로 돌린 결과 t2v 46.9 / 72.8 / 82.2 / MdR 2 / MnR 14.3, v2t 44.4 / 73.3 / 84.0 → 논문 값과 일치 `[측정: go_no_go.json]` (Proposal 그림 7과 같은 내용 → 한 줄로)
- **모델 선택 규칙** ⚠️: 최근 5회 평가 R@1 이동평균이 최고일 때 `model_best.pth` 저장. 이 평가가 **테스트셋**에서 이뤄짐 → 그래서 우리는 **best(공식 규칙)와 last(마지막 step) 둘 다** 보고
- **Colab 끊김 대응**: epoch마다 재개 상태 저장. seed 24는 3세션에 걸쳐 학습했고, 두 세션에서 겹쳐 평가된 7개 step의 지표가 모두 같았음(재개해도 결과 동일) `[측정: sessions_reconstructed.json]`
- 재현용 명령/노트북: `notebooks/01_train_baseline.ipynb` (seed 지정), 평가 `python -m tvr.evaluate` → 부록이나 코드 제출(Final)로 연결

#### 2.3 연산 환경과 실제 사용량 📊 표 4 (과제 필수 항목)
| run | GPU | 세션 수 | 총 학습 시간 | 최대 GPU 메모리(reserved) | step당 시간(중앙값) |
|---|---|---|---|---|---|
| seed 24 | A100-SXM4-80GB | 3 | **≥ 7.27h** (끊김·재학습 포함, 기록으로 복원한 하한) | 17.56GB | 0.689s |
| seed 0 | A100-SXM4-40GB | 1 | 6.13h | 17.66GB | 0.703s |
| seed 1 | A100-SXM4-40GB | 1 | 6.11h | 17.75GB | 0.702s |
- 출처: `experiments/results/reproduction_3seed.md`의 "Cost per run" 표, 각 `run_meta.json`, seed 24는 `sessions_reconstructed.json`
- 평가 1회 약 43초 × 50회 (학습 시간에 포함) `[측정: eval_*.json의 eval_seconds, go_no_go.json]`
- 데이터 로딩 대기 비율 0.06% 수준 → GPU 연산이 병목 `[측정: run_meta.json sec_per_step.data_wait_fraction]`
- 💭 [해석] "자원을 충분히 활용했는가"(과제 데이터 기준): 전체 9k + 공식 하이퍼파라미터 + seed 3개 + 매 평가 기록을 어떤 근거로 연결할지

#### 2.4 정량 결과 📊 표 5 (핵심 표)
출처: `experiments/results/reproduction_3seed.md`

**t2v (주 지표)**
| | R@1 | R@5 | R@10 | MdR | MnR |
|---|---|---|---|---|---|
| seed 24 (best = last) | 46.7 | 71.9 | 82.0 | 2 | 13.8 |
| seed 0 best / last | 46.5 / 46.0 | 71.2 / 71.1 | 82.1 / 81.9 | 2 / 2 | 14.0 / 14.0 |
| seed 1 best / last | 45.3 / 44.7 | 73.0 / 72.8 | 82.0 / 82.2 | 2 / 2 | 13.1 / 13.2 |
| **평균 ± 표준편차 (best)** | **46.17 ± 0.76** | 72.03 ± 0.91 | 82.03 ± 0.06 | 2.0 | 13.64 ± 0.45 |
| 평균 ± 표준편차 (last) | 45.80 ± 1.01 | 71.93 ± 0.85 | 82.03 ± 0.15 | 2.0 | 13.66 ± 0.39 |
| 공식 ckpt (우리 평가) = 논문 | 46.9 | 72.8 | 82.2 | 2 | 14.3 |

**v2t (보조)**: best 평균 44.63 ± 0.85 (45.5 / 44.6 / 43.8), 공식 ckpt 44.4
- 표준편차는 표본 표준편차(n=3) ⚠️ 표 아래에 명시
- 사실 재료:
  - 평균 R@1은 논문보다 0.73 낮고, 논문 값은 평균 + 표준편차 1개(46.93) 경계에 있음
  - R@10은 세 seed가 82.0~82.1로 거의 같고 R@1만 크게 흔들림(45.3~46.7)
  - seed 1은 R@1이 가장 낮지만 R@5(73.0)와 MnR(13.1)은 가장 좋음
  - best − last: 0.0 / 0.5 / 0.6 → 테스트셋으로 고르는 규칙이 R@1을 평균 약 0.37 올려줌
- 💭 [해석] "재현에 성공했다"고 말할 근거와 한계를 함께. 질문: 0.73 차이는 seed 편차로 설명되는가? 논문 46.9는 seed 몇 개의 결과인가(논문에 seed 수가 나오는지 ✅ 확인)? 테스트셋 선택이 논문 수치에도 들어 있다는 점은 어떤 의미인가?

#### 2.5 학습 곡선 📊 그림 1
- 파일: `experiments/results/figures/learning_curves.png` (생성 스크립트 `analysis/plot_curves.py`, 요약 수치 `learning_curves.json`)
- 내용: seed별 50회 평가의 R@1(옅은 선)과 5회 이동평균(진한 선), 논문 46.9 점선, epoch 경계
- 사실 재료:
  - 첫 평가(step 563) R@1 39.2~39.8 → epoch 1 끝(5,625) 43.5~44.2 `[eval_history.jsonl]`
  - 단일 평가 최고값 46.8 / 47.0 / 45.9, 이동평균 최고 46.66 / 46.36 / 45.36 `[learning_curves.json]`
  - seed 1은 epoch 4 이후 이동평균이 내려감(45.36 → last 44.7)
- 💭 [해석] 단일 평가 최고값(47.0)과 이동평균 기준 best의 차이 → 매 평가의 흔들림이 0.5~1점 수준이라는 점이 결과 해석에 주는 의미

#### 2.6 정성 결과 📊 그림 2~3 (과제 필수: "qualitative examples")
- 위치: Drive `results/xpool_9k_s24/analysis/gallery/`(가장 크게 실패한 12개), `gallery_near_miss/`(정답이 2위인 12개). 각 그림은 정답 영상과 오답 1위 영상의 12프레임 + 프레임별 attention 가중치 ✅ 그림을 직접 열어 고르기
- 추천 예시 (seed 24 기준, `analysis.json`, `gallery*/index.md`):
  - **실패 (hubness)**: "a man playing video games" — 정답 video9808이 **882위**, 1위는 video9679("the demonstration of mobile rack"). 같은 질의 문장이 3번 나오는데(정답 video9808/7461/7710) **세 번 모두 같은 오답**이 1위 → `q0803_video9808.png`
  - **실패 (일반적·짧은 질의)**: "a person is explaining something" — 정답 348위, 1위는 화장품 설명 영상 → `q0232_video7166.png`
  - **실패 (애니메이션/세부 구분)**: "cartoons are talking to each otehr"(질의 원문 오타 그대로) — 정답 312위 → `q0343_video7619.png`
  - **아슬아슬한 실패 (라벨 모호성)**: "a little boy singing in front of judges and crowd" — 정답 2위, 1위 캡션 "children singing a song as a group on a stage" → `q0008_video9778.png`
  - **아슬아슬한 실패 (비슷한 장면)**: "dogs and cats playing in a park" — 1위 캡션 "a dog and a cat are in a standoff" → `q0912_video7798.png`
  - **오답 캡션이 질의와 글자까지 같은 경우** (그림 없이 표로): "a group of people are dancing in a room" → 1위 캡션이 질의와 동일(정답 2위) `[analysis.json: ambiguous_failures]`
- ⚠️ **성공 예시**: 현재 갤러리는 실패 위주라 "정답 1위" 예시 그림이 없음. 선택지: ① near-miss를 "거의 맞힌 예"로 쓰기 ② 필요하면 `failure_gallery.py --select random`으로 추가 생성(노트북 02 셀 4 수정, 10분 내외) → 원하면 요청
- 💭 [해석] 예시마다 "attention이 어느 프레임에 몰렸는지 / 왜 1위 오답이 그럴듯한지"를 본인 관찰로 한 줄씩

#### 2.7 재현 결과 해석 (짧게)
- 💭 [해석] 아래 질문에 답하는 3~5줄
  - 3 seed 평균 46.17 ± 0.76은 논문 46.9와 비교해 "재현됨"으로 볼 수 있는가? 판단 기준(예: ±1.0 이내)을 먼저 밝히고 판정
  - best와 last 차이가 보여주는 평가 프로토콜의 한계
  - 남은 미세 차이 후보: 프레임 디코딩 라이브러리 버전, 토크나이저 정제(ftfy), GPU 비결정성 `[BASELINE_XPOOL.md §4]`

---

### 3. 한계 분석과 초기 아이디어 [3pt]

> 만점 조건: ① 재현 결과에 근거한 실패 분석(where + why) ② 한계 **하나**를 겨냥한 접근 ③ 예비 결과 ④ **본인 아이디어 vs 기존 연구 명시 + 인용**. 안 된 결과도 정직하게 쓰면 "genuine understanding"으로 읽힘.

#### 3.1 분석 방법 (도구 목록, 2~3줄)
- 모두 **학습 없이** 재현한 모델(seed 24, 일부는 seed 3개)의 1k-A 결과를 다시 계산
  - 질의 길이·카테고리·시간 표현별 성능, hubness 지표, 실패 질의와 오답 캡션의 유사도 → `analysis/analyze_dump.py` (seed 24)
  - 프레임 순서 probe(역순/셔플/가운데 1장) → 평가 시 프레임 순서만 바꿔 재평가 (seed 3개)
  - 같은 모델로 풀링 규칙만 바꾼 비교, 2단계 검색, 시간 측정 → `analysis/cost_tradeoff.py`, `analysis/pooling_budget.py` (seed 3개)
- 용어 정의: R@K, MdR/MnR(Proposal에 있음), **hubness**(소수 영상이 많은 질의의 상위권을 차지하는 현상), **N₁₀**(영상이 몇 개 질의의 top-10에 들어갔는지)

#### 3.2 어디서 실패하나 📊 표 6 (+ 그림 4 후보)
출처: `experiments/results/xpool_9k_s24/analysis/analysis.json` (seed 24), 순서 probe는 3 seed의 `eval_reversed/shuffled/middle.json`

| 관찰 | 수치 | 출처 |
|---|---|---|
| 짧은 질의에서 약함 | R@1: 6단어 이하 33.7 (n=273), 7–9단어 41.0, 10–12단어 54.1, 13단어 이상 66.3 (n=190) | analysis.json `per_query_length` |
| 카테고리별 격차 | 최저: 카테고리 1 → 27.8 (n=18), 2 → 26.4 (n=53) / 최고: 15 → 69.2 (n=13) | `per_category` |
| hubness | 영상 하나가 최대 8개 질의의 1위, 1,000개 중 338개는 한 번도 1위가 아님, N₁₀ 왜도 0.90 | `hubness` |
| hubness 진단 | 테스트 질의 전체로 재정규화(dual-softmax)하면 R@1 48.8, R@5 76.5, MnR 11.1 ⚠️ 테스트 질의를 함께 쓰는 방법이라 **성능 개선 방법이 아니라 진단용** | `diagnostic_dual_softmax_uses_test_queries` |
| 라벨 모호성 | 실패 533개 중 오답 영상 캡션과 질의의 코사인 유사도 평균 0.784, 0.9 이상이 15.8%, 0.95 이상이 3.9% | `failures` |
| 순서 무시 | 역순·셔플 결과가 원래와 **완전히 같음** (3 seed 모두: 46.7/46.5/45.3). 가운데 프레임 1장만 쓰면 29.9/30.0/29.3 | `eval_*.json` |
| 시간 표현 질의 | then/after/before 등 포함 질의(n=63) R@1 61.9 vs 미포함 45.7 | `temporal_words` |
| 질의 의존 풀링의 비용 | 질의 1,000개 순위 계산: 갤러리 1k에서 X-Pool 96ms vs 평균 풀링 0.23ms, 10k에서 866ms vs 0.67ms. 영상당 저장량 12배(프레임 12개 vs 벡터 1개). A100-SXM4-80GB | `xpool_9k_s24/analysis/cost/cost_tradeoff.json` |
| 이득이 고르지 않음 | 같은 모델에서 평균 풀링으로 바꾸면 R@1 42.0. 질의별: X-Pool만 맞힘 92, 평균만 맞힘 45, X-Pool이 순위를 낮춘 질의 205 | `cost_tradeoff.json: per_query_top1` |

- 그림 4 후보: `r1_by_category.png`, `hubness_n10.png`, `rank_hist.png` (모두 `experiments/results/xpool_9k_s24/analysis/`)
- ⚠️ 카테고리 이름: json에는 **번호만** 있음. 이름을 쓰려면 "캡션 상위 단어로 추정"이라고 밝힐 것(예: 2번 상위 단어 game, video, playing, minecraft) `[DATASET_FACTSHEET_TABLES.md]`
- ⚠️ 표본 크기: 카테고리 1(n=18), 15(n=13)처럼 작은 그룹은 해석을 약하게

#### 3.3 왜 실패하나: 원인별 정리
- **(a) hubness** — 재료: 같은 오답이 여러 질의에서 반복(예시 "video games" ×3), 진단용 재정규화로 R@1 +2.1. 3 seed 비교에서 평균 풀링이 hubness가 더 심함(최대 1위 횟수 12.3 vs X-Pool 7.0, 왜도 1.04 vs 0.89) `[pooling_budget/summary.json: hubness]`
  - 💭 [해석] 왜 특정 영상이 hub가 되는지 갤러리를 보고 가설 하나(예: 일반적인 장면, 화면 녹화, 텍스트 화면 등)
- **(b) 짧고 일반적인 질의 + 라벨 모호성** — 재료: 짧은 질의 33.7 vs 긴 질의 66.3, 실패의 15.8%가 질의와 거의 같은 캡션의 다른 영상. 1k-A는 질의마다 정답이 하나뿐이라, 내용상 맞는 다른 영상도 오답 처리됨
  - 💭 [해석] 이 중 "모델의 한계"와 "평가 방식의 한계"를 구분
- **(c) 시간 순서** — 재료: 풀링이 프레임 순서에 대해 구조적으로 불변(역순·셔플 결과가 정확히 같음). 그러나 시간 표현 질의가 오히려 더 잘 맞음(61.9)
  - 💭 [해석] "구조적 한계는 확실하지만 MSR-VTT 지표로는 손해가 드러나지 않는다"를 어떻게 해석할지(벤치마크가 순서를 요구하지 않는가?)
- **(d) 비용과 불균형한 이득** (교수님 피드백) — 재료: X-Pool은 질의마다 모든 영상의 표현을 다시 계산해야 해서 미리 인덱싱할 수 없음. 평균 풀링보다 R@1 +4.7(같은 encoder)이지만 이득이 모든 질의에 고르지 않음(평균만 맞힌 질의 45개)

#### 3.4 다룰 한계 선정 (한 가지)
- 선택: **(d) 질의 의존 풀링의 계산 비용과, 질의마다 고르지 않은 이득**
- 선정 근거(재료):
  - 교수님 피드백이 직접 지목한 trade-off
  - 측정 수치가 가장 분명함(10k 갤러리에서 866ms vs 0.67ms)
  - 학습 없이 기존 모델로 바로 실험 가능 → 예비 결과를 seed 3개로 낼 수 있음
- 보조로 (a) hubness에 대한 기존 기법(QB-Norm)도 시험 → 3.6에서 결과와 함께
- 💭 [해석] 이 선택이 왜 "이 베이스라인의 핵심"을 건드리는지 한두 줄(X-Pool의 기여가 바로 질의 의존 풀링이므로)

#### 3.5 접근 A (본인 아이디어): U-Pool 📊 표 7, 그림 5
**정의 (재료)**
- X-Pool 풀링 모듈 = ① 질의로 프레임 가중치를 정하는 attention(q, k 투영) + ② 가중합을 다듬는 학습된 층(v 투영, 출력 투영, LayerNorm, FC+잔차)
- U-Pool = 학습된 X-Pool 모듈에서 attention 가중치만 **모든 프레임 1/12로 고정**. 나머지 학습된 층은 그대로 사용, **추가 학습 없음**
- 결과: 질의와 무관한 영상 벡터 1개 → 평균 풀링처럼 미리 계산해 인덱스로 저장 가능
- 수식 재료(본인 표기로): X-Pool은 영상 표현이 `z_{v|t} = g( Σ_f a_f(t) · W_v·LN(x_f) )`, U-Pool은 `a_f = 1/F`로 둔 `z_v = g( (1/F) Σ_f W_v·LN(x_f) )` (g = 출력 투영 → LN → FC 잔차 → LN). ✅ `third_party/xpool/modules/transformer.py`의 forward와 기호를 맞출 것
- 검증: α=0(균일)일 때 공식 수식으로 계산한 점수와 U-Pool 점수 차이 ≤ 1.6×10⁻⁷, α=1일 때 공식 X-Pool과 차이 ≤ 6.7×10⁻⁸ `[pooling_budget/xpool_9k_s*.json: sanity_*]`

**결과 1: X-Pool의 이득은 어디서 오나** (3 seed 평균 ± 표준편차, `experiments/results/pooling_budget/pooling_budget_report.md` §1)
| 점수 규칙 | 질의 의존 | 영상당 저장 | R@1 | R@5 | R@10 | MnR |
|---|---|---|---|---|---|---|
| 평균 풀링 | X | 벡터 1개 | 41.8 ± 0.6 | 68.5 ± 0.5 | 78.8 ± 0.8 | 17.4 ± 0.3 |
| **U-Pool** | X | 벡터 1개 | **43.2 ± 0.2** | 68.8 ± 0.7 | 79.4 ± 1.0 | 15.8 ± 0.4 |
| X-Pool | O | 프레임 12개 | 46.2 ± 0.8 | 72.0 ± 0.9 | 82.0 ± 0.1 | 13.6 ± 0.5 |
- 평균 → X-Pool 이득 4.4점 = 학습된 층 몫 1.4 (약 30%) + 질의 조건 attention 몫 3.0 (약 70%)
- 질의 단위(1,000개 중, 3 seed 평균): 학습된 층이 고친 질의 49.3 / 망친 질의 35.3, attention이 고친 질의 67.0 / 망친 질의 37.0 `[summary.json: top1_transitions]`
- ⚠️ 여기서 X-Pool 46.2는 seed별 dump의 점수로 다시 계산한 값(평균 46.17과 반올림 차이). 2.4의 46.17과 같은 값임을 한 번 언급

**결과 2: attention의 날카로움** 📊 그림 5 = `pooling_budget/r1_vs_alpha.png`
- attention logit에 α를 곱해 0(=U-Pool) ~ 2까지 변화
- R@1: α 0 → 43.2, 0.25 → 44.5, 0.5 → 45.8, 0.75 → 46.1, **1 → 46.2(최고)**, 1.5 → 45.0, 2 → 44.0
- R@10은 α=1.5에서 82.4로 α=1(82.0)보다 약간 높음
- ⚠️ α는 테스트셋으로 고르면 안 됨 → **개선 방법이 아니라 분석 결과**로만 서술
- 💭 [해석] 학습된 X-Pool의 attention 세기가 R@1 기준으로 이미 최적 근처라는 점, R@10은 더 날카로울 때 오른다는 점의 의미

**결과 3: U-Pool을 2단계 검색의 1단계로** → 3.6 표 8에서 기존 방법과 같이 비교

**출처 구분 (본문에 명시)** ⚠️
- **본인 아이디어로 제시**하되 범위를 좁혀서: "학습된 X-Pool 모듈을 균일 attention으로 바꿔 **질의 무관 인덱스로 재사용**하고, 이를 통해 이득을 분해하고 2단계 검색의 1단계로 쓰는 것"
- 함께 밝힐 점: 균일 가중치 비교 자체는 흔한 ablation 형태이고, 질의 무관 1단계 + 질의 조건 재정렬 구조는 X-Pool 부록 C와 Bain et al. 2022에 있음 → "to our knowledge" 수준으로
- 확인한 범위(문헌 조사): X-Pool 본문·부록, Bain et al. 2022, EERCF, ProCLIP, Prompt Switch, TeachCLIP, QB-Norm, ranked-list truncation 계열 → 같은 방식 없음 `[docs/IDEAS.md §2-1]`

#### 3.6 접근 B (기존 연구 적용): 2단계 검색, 적응형 깊이, QB-Norm, 점수 융합 📊 표 8, 그림 6
출처: `experiments/results/pooling_budget/pooling_budget_report.md` §4~6, 3 seed 평균 ± 표준편차

**B-1. 고정 P 2단계 검색** (기존: X-Pool 부록 C, Bain et al. 2022)
- 방법: 질의 무관 점수로 상위 P개 → 그 P개만 X-Pool로 재정렬. "예산" = 질의당 X-Pool 계산 쌍 수
- 결과 (E1 = 평균 풀링 1단계 / O1 = U-Pool 1단계, R@1):

| 예산 P | E1 (평균 → X-Pool) | O1 (U-Pool → X-Pool) |
|---|---|---|
| 1 | 41.8 ± 0.6 | 43.2 ± 0.2 |
| 2 | 44.4 ± 0.6 | 44.6 ± 0.4 |
| 3 | 44.7 ± 0.2 | 45.1 ± 0.3 |
| 5 | 45.2 ± 0.6 | 45.4 ± 0.6 |
| 10 | 45.9 ± 0.8 | 45.8 ± 1.0 |
| 20 | 46.2 ± 1.0 (MnR 16.9) | 46.2 ± 1.0 (MnR 15.5) |
| 전체 1,000 | 46.2 (X-Pool) | 46.2 |
- 사실 재료:
  - **P=20(전체의 2%)에서 X-Pool 전체 계산과 같은 R@1**
  - O1이 작은 예산에서 0.2~0.4 높지만 **seed 표준편차(≈0.6) 안** → 개선이라 주장하지 않기 ⚠️
  - MnR은 O1이 모든 예산에서 낮음(좋음): P=20에서 15.5 vs 16.9, P=100에서 14.7 vs 15.9 → 후보에 못 든 정답의 순위를 더 나은 1단계가 정하기 때문
- 시간 (A100-SXM4-80GB, 질의 1,000개, seed 24 모델) `[pooling_budget_report.md §5]`:
  - 갤러리 1k: X-Pool 전체 87.3ms, 2단계 P=5 6.3ms, P=20 6.4ms
  - 갤러리 10k: X-Pool 전체 866.5ms, 2단계 P=5 10.6ms, P=20 **14.1ms** (약 61배 빠름), U-Pool 인덱스 생성(오프라인) 8.5ms
  - ⚠️ 1k X-Pool 시간이 3.2의 96ms와 다른 것은 별도 측정이기 때문. 한 표 안에서는 한 출처만 쓰기
- 그림 6 = `pooling_budget/r1_vs_budget.png` (가로축 예산 로그 스케일, 오차막대 = seed 표준편차)

**B-2. 질의별 적응형 재정렬 깊이** (기존 아이디어 적용: ranked-list truncation, Lien et al. 2019 등)
- 방법: 1단계 점수의 1위−2위 차이가 작은(덜 확신하는) 질의에 큰 P(50), 나머지에 작은 P(0 또는 5). 평균 예산은 **라벨 없이** 1단계 점수만으로 맞춤
- 결과: **같은 예산의 고정 P보다 나쁨**. 예산 5: E2(평균, P 0/50) 42.2 ± 0.8 vs E1 45.2 ± 0.6. 예산 20: 44.4 vs 46.2. U-Pool 1단계(O2)는 E2보다 낫지만(예산 5에서 43.5) 고정 P보다는 여전히 낮음
- 💭 [해석] 왜 안 됐는가: 1단계 점수의 확신도가 "X-Pool이 필요한 질의"를 가려내지 못함 → 확신이 높은 질의의 1위가 틀린 경우가 많다는 뜻. 더 나은 신호(예: 학습된 판별기)는 Final 후보

**B-3. QB-Norm** (기존: Bogolin et al., CVPR 2022) — hubness 겨냥
- 설정: 9k **학습** 캡션 5,000개를 querybank로(테스트 영상 제외 확인), 논문 기본값 β=20, k=1. **우리 테스트셋으로 조정하지 않음**
- 결과: hubness 지표는 크게 줄었으나(N₁₀ 왜도 0.89 → 0.55) **R@1은 46.2 → 45.2 ± 0.9로 하락**, R@10 82.0 → 82.1
- 민감도 참고(주장 X): β=40에서 R@1 45.9, R@10 83.2
- 관찰: 질의의 약 97%에 정규화가 적용됨(DIS의 활성 집합이 갤러리 대부분을 덮음) `[pooling_budget/xpool_9k_s*.json: frac_queries_normalised]`
- 💭 [해석] 왜 안 됐는가: 원래 X-Pool의 hubness가 평균 풀링보다 이미 약하다는 점(3.3-a), 활성 집합이 너무 넓어 "hub만 골라 보정"하는 장치가 작동하지 않았다는 점, 진단용 dual-softmax의 +2.1은 테스트 질의를 써서 과대평가였을 가능성

**B-4. 점수 융합** (일반적 앙상블, 독창성 주장 X)
- 설정: X-Pool 점수 + λ × (U-Pool 또는 평균 풀링 점수), λ=1 사전 고정
- 결과: λ=1에서 R@1 45.3 ± 0.4 (U-Pool), 45.1 ± 0.6 (평균) → 하락. λ를 줄이면 X-Pool과 비슷할 뿐 오르지 않음(λ=0.25: 46.0 / 46.2)
- 💭 [해석] 질의별로 둘 중 맞는 쪽을 고를 수 있다면 정답 1위가 1,000개 중 520개(52.0%)까지 가능하지만 `[summary.json: oracle_any_of_three]`, 단순 선형 융합으로는 그 상보성을 쓰지 못함 → 질의별로 고르는 장치가 필요

#### 3.7 예비 결론과 Final 계획
- 재료 정리(되는 것 / 안 되는 것):
  - ✅ X-Pool 이득의 약 70%가 질의 조건 attention, 30%가 학습된 층 (U-Pool 분해)
  - ✅ 2단계 검색으로 X-Pool 계산의 2%만 써도 같은 R@1, 10k 갤러리에서 약 61배 빠름
  - ✅ U-Pool은 평균 풀링과 같은 비용으로 R@1 +1.4, MnR −1.6 → 더 나은 1단계 / 인덱스
  - ❌ U-Pool 1단계의 R@1 이득은 seed 편차 안
  - ❌ 확신도 기반 적응형 깊이, QB-Norm, 단순 점수 융합은 R@1을 올리지 못함
- Final 방향 후보 (과제: Final에서는 **베이스라인보다 나은 방법 필수**):
  - 학습 없는 후처리로는 R@1이 오르지 않았으므로 **학습 단계**에서 다루기
  - 후보 1: 질의별로 "X-Pool이 필요한지/어느 점수를 믿을지"를 학습하는 장치(oracle 52.0이 보여준 상보성 활용)
  - 후보 2: 학습 중 hubness 억제 또는 캡션이 거의 같은 영상을 오답으로 밀어내지 않는 손실(라벨 모호성 15.8%)
  - 검증 방식: seed 3개, best와 last 모두 보고, 표준편차보다 큰 개선만 주장
- 💭 [해석] Final 후보 중 어느 쪽이 3장의 분석과 가장 직접 연결되는지 본인 판단으로 한 줄. Final 방향은 아직 확정 전이므로 "후보"로 쓰기

#### 3.8 출처 구분 📊 표 9 (과제 필수 요구)
| 접근 | 구분 | 인용 |
|---|---|---|
| U-Pool (균일 attention으로 질의 무관 인덱스 재사용, 이득 분해, 1단계 사용) | **본인 아이디어** (조사 범위 내에서 동일 사례 없음) | 관련: X-Pool [1] 부록 C, Bain et al. [9] |
| attention 날카로움(α) 분석 | 본인 분석 | – |
| 고정 P 2단계 검색 | 기존 연구 적용 | X-Pool [1] 부록 C, Bain et al. [9] |
| 질의별 적응형 깊이 | 기존 아이디어 적용 | Lien et al. [11], Bahri et al. [12] 등 |
| QB-Norm (DIS) | 기존 연구 적용 | Bogolin et al. [10] |
| 점수 융합 | 일반 기법 (독창성 주장 없음) | – |
| hubness 진단(dual-softmax) | 기존 기법, 진단용 | Cheng et al. (CAMoE) ✅ 서지 확인 |

---

### 참고문헌 (Proposal [1]~[8]에 이어서 추가)
- [9] M. Bain, A. Nagrani, G. Varol, A. Zisserman, "A CLIP-Hitchhiker's Guide to Long Video Retrieval," arXiv:2205.08508, 2022 ✅ 저자 확인
- [10] S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, S. Albanie, "Cross Modal Retrieval with Querybank Normalisation," CVPR, pp. 5194–5205, 2022
- [11] Y.-C. Lien, D. Cohen, W. B. Croft, "An Assumption-Free Approach to the Dynamic Truncation of Ranked Lists," ICTIR, 2019
- [12] D. Bahri, Y. Tay, C. Zheng, D. Metzler, A. Tomkins, "Choppy: Cut Transformer for Ranked List Truncation," SIGIR, 2020
- (선택) X. Cheng et al., "Improving Video-Text Retrieval by Multi-Stream Corpus Alignment and Dual Softmax Loss," arXiv:2109.04290, 2021 (CAMoE, dual softmax) ✅ 확인
- (선택) M. Radovanović, A. Nanopoulos, M. Ivanović, "Hubs in Space," JMLR, 2010 (hubness 개념) ✅ 확인
- ⚠️ 본문에서 수치를 인용할 때마다 번호를 붙이고, 참고문헌 형식을 Proposal과 통일

### AI 도구 사용 고지 (짧게, 본인 문장으로)
- 사실 재료: 코드 작성·디버깅과 실험 자동화, 논문 내용 이해 보조, 결과 표·그림 생성 스크립트 작성에 AI 도구 사용. 보고서 문장은 직접 작성
- ⚠️ 실제 사용 범위와 일치하게(Proposal 때 지적했던 부분)

---

## 3. 표·그림 목록 (넣는 순서대로)

| 번호 | 내용 | 파일/출처 | 필수? |
|---|---|---|---|
| 표 1 | 데이터셋 분할 비교 | Proposal 표 1 + `dataset_stats.json` | ○ |
| 표 2 | 관련 연구 요약 | Proposal 표 "2.5" 축약 | ○ |
| 표 3 | 재현 설정 | 위 §2.1 | **필수** |
| 표 4 | 연산 환경·실제 사용량 | `reproduction_3seed.md` Cost per run | **필수** |
| 표 5 | 3 seed 정량 결과 | `reproduction_3seed.md` | **필수** |
| 그림 1 | 학습 곡선 | `experiments/results/figures/learning_curves.png` | 권장 |
| 그림 2~3 | 정성 예시(실패, 아슬아슬한 실패) | Drive `results/xpool_9k_s24/analysis/gallery*/` | **필수** |
| 표 6 | 실패 분석 요약 | §3.2 표 | **필수** |
| 그림 4 | 카테고리별 R@1 또는 hubness 분포 | `xpool_9k_s24/analysis/*.png` | 선택 |
| 표 7 | 평균 / U-Pool / X-Pool 비교 | `pooling_budget_report.md` §1 | **필수** |
| 그림 5 | α 곡선 | `pooling_budget/r1_vs_alpha.png` | 권장 |
| 표 8 | 2단계·QB-Norm·융합 결과 (+시간) | `pooling_budget_report.md` §4~6 | **필수** |
| 그림 6 | 예산-정확도 곡선 | `pooling_budget/r1_vs_budget.png` | **필수** |
| 표 9 | 본인 vs 기존 출처 구분 | §3.8 | **필수** |

- 그림 캡션에는 seed 수, 오차막대 의미(표본 표준편차), GPU를 적기
- 그림 속 영어 라벨(E1, O1 등)을 본문 용어와 맞추기 → 표 8 범례로 대응 관계 설명

---

## 4. 작성 팁 (점수 관점)

### 4-1. 배점 항목별 "만점 조건" 체크
- **재현 2점**: 정량(표 5) / 정성(그림 2~3) / 세부사항(표 3 + 2.2) / 연산 사용량(표 4) — 네 가지 모두 있는가
- **한계·아이디어 3점**: where(표 6) / why(3.3) / 한계 하나 선정 근거(3.4) / 접근 + 예비 결과(3.5~3.6) / 출처 구분(표 9)

### 4-2. 감점을 부르는 실수 ⚠️
- 표준편차 안의 차이를 "개선"이라 쓰기 (O1의 +0.2~0.4, 융합 λ=0.25의 46.2 등)
- 테스트셋으로 고른 값(α=1.5, β=40)을 결과처럼 쓰기 → "민감도 참고"로만
- dual-softmax 48.8을 "가능한 성능"처럼 쓰기 → 테스트 질의를 함께 쓰는 진단값
- U-Pool을 "세계 최초"처럼 단정 → "조사 범위에서 같은 사례를 찾지 못함"
- 카테고리 번호에 이름을 확정적으로 붙이기
- 3.2의 1k X-Pool 시간 96ms와 3.6의 87.3ms를 같은 측정처럼 섞기
- GPU를 "A100 80GB"로만 쓰기 (seed 0·1은 40GB)
- seed 24 학습 시간을 6.1h로 쓰기 (실제 ≥7.27h, 3세션)

### 4-3. 문장을 쓸 때
- 수치 → 해석 순서. 해석에는 "~로 보인다", "~를 시사한다"처럼 근거의 세기에 맞는 표현
- "안 된 결과"는 원인 가설과 함께 쓰면 분석력으로 읽힘(과제 Final 요구사항도 "안 된 시도 포함")
- 3장 첫머리에 교수님 피드백을 한 줄로 받아서 시작하면 흐름이 자연스러움

### 4-4. 용어 통일표 (첫 등장 시 영문 병기)
| 한국어 | 영문/약어 | 메모 |
|---|---|---|
| 텍스트 조건부 풀링 / 질의 의존 풀링 | text-conditioned pooling | 둘 중 하나로 통일 |
| 평균 풀링 | mean pooling | |
| U-Pool | uniform-attention X-Pool | 처음에 정의 한 번, 이후 U-Pool |
| 2단계 검색 | two-stage retrieval (shortlist + re-ranking) | |
| 예산 | X-Pool 계산 쌍 수 / 질의 | 정의를 표 8 위에 |
| 허브 현상 | hubness | |
| 마지막 체크포인트 / 공식 선택 체크포인트 | last / best-on-test | |
| 표준편차 | sample std (n=3) | |

### 4-5. 숫자 표기
- R@K는 % 단위, 소수점 1자리(평균 ± 표준편차는 표 5만 2자리, 나머지는 1자리로 통일 권장)
- 시간은 ms/h 단위 명시, GPU 이름 함께

---

## 5. 제출 전 체크리스트
- [ ] 표지에 "Intermediate Progress Report"
- [ ] Proposal 요약 + 변경 사항(GPU 정정, CLIP4Clip 표현 수정, 피드백 반영)
- [ ] 표 3·4·5 (설정, 사용량, 3 seed 결과) 모두 있음
- [ ] 정성 예시 그림 2개 이상, 캡션 있음
- [ ] 한계 "하나"를 명시적으로 선정했고 근거가 있음
- [ ] 본인 아이디어 vs 기존 연구 표(표 9)와 인용 번호
- [ ] 안 된 결과(적응형 깊이, QB-Norm, 융합)도 원인 가설과 함께 포함
- [ ] 표준편차 안의 차이를 개선으로 쓰지 않음
- [ ] 테스트셋 조정값(α, β 민감도)을 결과로 쓰지 않음
- [ ] 모든 수치에 출처 파일과 일치하는지 확인(아래 부록 A)
- [ ] 참고문헌 [9]~[12] 추가, 형식 통일
- [ ] AI 사용 고지가 실제 사용 범위와 일치
- [ ] PDF로 변환 후 그림 깨짐 확인

---

## 부록 A. 근거 파일 위치
| 내용 | 파일 |
|---|---|
| 3 seed 재현 표 | `experiments/results/reproduction_3seed.md` (`.json`) |
| seed별 결과·환경·시간 | `experiments/results/xpool_9k_s{24,0,1}/` (`run_meta.json`, `eval_*.json`, `eval_history.jsonl`, `train.log`) |
| seed 24 세션 복원 | `experiments/results/xpool_9k_s24/sessions_reconstructed.json` |
| 학습 곡선 | `experiments/results/figures/learning_curves.png` (`.json`) |
| 실패 분석 (seed 24) | `experiments/results/xpool_9k_s24/analysis/analysis.json`, `analysis_report.md`, 그림 3개 |
| 비용 분석 (seed 24) | `experiments/results/xpool_9k_s24/analysis/cost/` |
| U-Pool·2단계·QB-Norm·융합 (3 seed) | `experiments/results/pooling_budget/` (`summary.json`, `pooling_budget_report.md`, 그림 2개) |
| 갤러리 그림 | Drive `컴비 프로젝트/results/xpool_9k_s24/analysis/gallery/`, `gallery_near_miss/` |
| 공식 ckpt 사전 검증 | `experiments/results/go_no_go.json` |
| 데이터셋 통계 | `experiments/results/dataset_stats.json`, `docs/DATASET_FACTSHEET_TABLES.md` |
| 문헌 확인 기록 | `docs/IDEAS.md` §2-1 |
| 실험 기록 | `experiments/LOG.md` |

## 부록 B. 코드 위치 (Final의 코드 제출과도 연결)
- 학습/평가 래퍼: `src/tvr/` (`train.py`, `evaluate.py`, `compat.py`, `frame_cache.py`)
- 분석: `analysis/analyze_dump.py`, `failure_gallery.py`, `cost_tradeoff.py`, `pooling_budget.py`, `embed_querybank.py`, `seed_summary.py`, `plot_curves.py`
- 노트북: `notebooks/01_train_baseline.ipynb`, `02_analysis.ipynb`, `03_pooling_budget.ipynb`
- 테스트: `PYTHONPATH=src python -m pytest -q tests/`
