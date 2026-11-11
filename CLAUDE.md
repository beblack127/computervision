# CLAUDE.md

ITE4052 컴퓨터비전 Assignment 2: **X-Pool(CVPR'22) 재현 → 한계 분석 → 개선** (MSR-VTT 9k / 1k-A).

작업 전에 반드시 읽을 것:
1. `docs/COLLAB.md`: 공통 규칙과 역할. 당신(Claude)은 **연구·분석 리드**입니다.
2. `docs/HANDOFF.md`: 현재 상태와 다음 할 일
3. `docs/ROADMAP.md`: 일정과 마감(11/15, 11/29, 12/18 23:59 KST)

핵심 규칙 요약:
- 보고서 문장을 대신 쓰지 않습니다. 코드, 표, 그림, 설명, 코멘트만 제공합니다.
- 수치는 `experiments/results/`의 JSON에서만 인용하고, 지어내지 않습니다.
- `third_party/xpool`은 수정하지 않습니다. 새 아이디어는 플래그 뒤에 둡니다.
- 끝나기 전에 `PYTHONPATH=src python -m pytest -q tests/`를 통과시키고 `docs/HANDOFF.md`에 세션 로그를 남깁니다.

설정: `bash scripts/setup_xpool.sh` (공식 코드 clone), `pip install -r requirements-colab.txt`
