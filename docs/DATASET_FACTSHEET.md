# 데이터셋 팩트 시트 (Proposal "Topic & Dataset" 준비용)

> 과제 규정: 데이터셋 항목은 **자세히 쓰지 않으면 0점**입니다. 크기, 출처, 라이선스, 전처리, train/val/test 분할 기준, 샘플링 여부와 근거, 연산 환경과 실제 사용량이 필요합니다.
> 아래는 **채워야 할 항목 체크리스트와 근거 자료 위치**입니다. 문장은 직접 쓰세요.

## 체크리스트 → 근거

| 과제 요구 항목 | 근거 / 값 위치 | 상태 |
|---|---|---|
| 데이터셋 이름·출처 | MSR-VTT (Xu et al., CVPR 2016), 영상: Frozen-in-Time 미러(CLIP4Clip/X-Pool README가 안내), 분할: X-Pool 공식 repo 동봉 파일 | ✅ |
| 규모 | 영상 10,000 / 캡션 200,000 / 총 41.7시간 / 클립 10–30초 (`DATASET_FACTSHEET_TABLES.md`) | ✅ (계산됨) |
| 분할 기준 | 학습 9k(9,000영상, 180,000쌍), 테스트 1k-A(1,000쌍). 출처와 구성은 TABLES 참고. 학습과 테스트 겹침 0 | ✅ |
| validation 부재와 모델 선택 문제 | `BASELINE_XPOOL.md` §3 | ✅ (본인 해석 필요) |
| 전처리 | 영상당 12프레임 uniform 샘플링 → 학습: RandomResizedCrop(224)+flip / 테스트: Resize+CenterCrop, CLIP 정규화. 텍스트: CLIP BPE 토크나이저(최대 77 토큰) | ✅ |
| 샘플링(데이터 축소) 여부 | **축소 없음(전체 9k + 1k-A)**. 단, go/no-go 결과로 확정. 시간이 부족하면 대안은 논문이 직접 보고한 7k 설정(43.9) | ⏳ go/no-go 후 확정 |
| 라이선스/이용 조건 | MSR-VTT는 연구 목적 공개 데이터셋. 원본은 YouTube 클립이라 저작권은 원 게시자에게 있음. **공식 페이지에서 직접 확인 후 기재** (https://msrvtt.github.io/ 등) | ⚠️ 직접 확인 필요 |
| 연산 환경 | Colab A100. GPU 이름, 메모리, CPU, RAM은 `experiments/results/go_no_go.json` | ⏳ go/no-go 후 |
| 실제 사용량 | batch 32, 5 epoch, 피크 메모리, 총 학습시간 → 각 run의 `run_meta.json` | ⏳ 재현 후 (Intermediate) |
| "자원을 충분히 썼는가" 논리 | 전체 데이터 + 공식 하이퍼파라미터 + seed 3개 + 50회 평가 기록 | 📝 본인 서술 |

## 직접 확인해서 쓸 만한 관찰 (계산 결과에서 나온 사실)

- 9k 학습셋 9,000개 중 1,990개는 원래 2016 공식 **test split**에서 왔습니다. 즉, 이 검색 프로토콜은 2016 원래 분할과 다릅니다.
- 1k-A 질의 1,000개 중 965개는 해당 영상의 MSR-VTT 캡션과 글자까지 같고, 35개는 다릅니다(JSFusion 쪽 전처리 차이로 추정. 추정임을 명시할 것).
- 테스트 질의 평균 9.4단어, 어휘 2,002개. 시간어(then/after/before/while 등)가 포함된 질의는 63개 → H1 분석 대상입니다.
- `MSRVTT_data.json`의 `split` 필드는 모든 영상이 `train`으로 표시되어 있어 쓸모가 없습니다. 원래 분할은 id 범위로 계산했습니다(train 0–6512, validate 6513–7009, test 7010–9999).
- 카테고리 20개는 json에 id만 있습니다. TABLES의 "top words"는 캡션에서 뽑은 **참고용 힌트**이고 공식 이름이 아닙니다.

재생성: `python scripts/dataset_stats.py` → `experiments/results/dataset_stats.json`, `docs/DATASET_FACTSHEET_TABLES.md`
