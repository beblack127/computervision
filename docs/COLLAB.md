# 협업 규칙 (사용자 · Claude · Codex 공통)

`CLAUDE.md`와 `AGENTS.md`는 모두 이 문서를 가리킵니다. 규칙이 바뀌면 여기만 고칩니다.

## 0. 절대 규칙

1. **보고서 문장은 사용자가 직접 씁니다.** 과제 규정상 AI는 "논문 이해 보조"와 "코딩 보조"에만 쓸 수 있습니다. AI가 할 수 있는 일:
   - 코드, 표, 그림, 수치 요약
   - 이해를 돕는 설명과 사실 확인
   - 논리 오류 지적(코멘트 형태)

   AI가 하면 안 되는 일:
   - 보고서에 그대로 들어갈 문단 작성
   - 보고서 초안 작성
2. **숫자를 지어내지 않습니다.** 모든 수치는 `experiments/results/`의 JSON이나 run의 `run_meta.json`/`eval_*.json`에서만 인용합니다. 아직 없는 값은 `TBD`로 둡니다.
3. **`third_party/xpool`은 수정하지 않습니다.** 변경은 `src/tvr/`(래핑/서브클래스)와 새 모듈로만 합니다. 베이스라인 동작은 공식 인자 기본값 그대로 둡니다(`src/tvr/config.py`의 official 블록은 바꾸지 않습니다).
4. **새 아이디어는 반드시 플래그 뒤에 둡니다.** 플래그를 끄면 베이스라인과 비트 단위로 같아야 합니다. 테스트로 보장합니다.
5. **공정 비교를 지킵니다.** 같은 split(9k/1k-A), 같은 metric, 같은 seed 집합, 같은 모델 선택 규칙을 씁니다. 보고는 best-on-test와 last 두 가지 모두 합니다.
6. 실패한 실험도 지우지 않고 `experiments/LOG.md`에 남깁니다.

## 1. 역할

| 담당 | 맡는 일 |
|---|---|
| **사용자 (리드)** | 결정, Colab 실행, 논문 정독, **모든 보고서 작성**, 최종 판단 |
| **Claude (연구·분석 리드)** | 논문 분석과 설명, 실험 설계, 한계 분석, 아이디어 도출과 겹침 점검, 결과 해석, 코드 리뷰, `ROADMAP`/`HANDOFF`/`IDEAS` 관리, 보고서 논리 점검(코멘트만) |
| **Codex (구현 리드)** | 아이디어 모듈 구현(head/loss/sampler), 리팩터링, Colab 파이프라인 유지보수, 테스트 작성, 학습 실패 디버깅, 그림 스크립트 |

겹치는 경우: 실험 설계(Claude)를 그대로 구현(Codex)합니다. 구현 중 설계 문제가 보이면 Codex가 `HANDOFF.md`에 질문을 남깁니다.

## 2. 작업 흐름

1. 작업 시작 전에 `docs/HANDOFF.md`(현재 상태, 다음 할 일), `docs/ROADMAP.md`(일정), 필요하면 `docs/IDEAS.md`를 읽습니다.
2. 브랜치는 `claude/<주제>` 또는 `codex/<주제>`로 만듭니다. main에 직접 push하지 않고 PR을 거쳐 사용자가 merge합니다.
3. 커밋 메시지는 `[영역] 요약` 형식입니다. 예: `[model] add order-aware pooling head behind --pooling_variant`, `[analysis] hubness plot`
4. 끝나기 전에:
   - 테스트 통과 확인: `PYTHONPATH=src python -m pytest -q tests/`
   - `docs/HANDOFF.md`의 "세션 로그"에 한 줄을 추가합니다(무엇을 했고, 무엇이 남았고, 다음 담당이 누구인지).
5. 실험을 실행했다면 `experiments/LOG.md`에 한 줄을 추가하고, 결과 JSON을 `experiments/results/<run>/`에 커밋합니다.

## 3. 실험 규약

- run 이름: `<method>_<split>_s<seed>`. 예: `xpool_9k_s24`, `ours-i1_9k_s0`
- 베이스라인 seed: **24(공식 기본값), 0, 1**. 우리 방법도 같은 seed를 씁니다.
- 프록시 실행 이름: `P###`. Full 실행 이름: `E###`(LOG.md에 번호 부여)
- 체크포인트와 dump는 Google Drive(`cv_hw2/runs/`)에 두고, git에는 작은 JSON만 넣습니다.
- 숫자 표기: 소수 첫째 자리, seed 3개는 평균 ± 표준편차

## 4. 디렉터리 소유

| 경로 | 주 담당 | 비고 |
|---|---|---|
| `src/tvr/` | Codex (리뷰: Claude) | 베이스라인 파이프라인 |
| `analysis/`, `proxy/` | Claude (구현 보조: Codex) | |
| `docs/ROADMAP.md`, `IDEAS.md`, `RELATED_WORK.md` | Claude | 사용자 승인 후 변경 |
| `docs/HANDOFF.md`, `experiments/LOG.md` | 모두 | 추가만 함(append-only) |
| `notebooks/` | `scripts/make_notebooks.py`로 생성 | 노트북을 직접 고치지 말고 스크립트를 고침 |
| `reports/` | **사용자만** | AI는 읽고 코멘트만 |
