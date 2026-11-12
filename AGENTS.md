# AGENTS.md

ITE4052 컴퓨터비전 Assignment 2: **X-Pool(CVPR'22) 재현 → 한계 분석 → 개선** (MSR-VTT 9k / 1k-A).

작업 전에 반드시 읽을 것:
1. `docs/COLLAB.md`: 공통 규칙과 역할. 당신(Codex)은 **구현 리드**입니다.
2. `docs/HANDOFF.md`: 현재 상태와 다음 할 일 (Codex 담당 항목 확인)
3. `docs/BASELINE_XPOOL.md`: 공식 코드 구조와 호환성 수정 사항

핵심 규칙 요약:
- `third_party/xpool`(공식 코드, 고정 커밋)은 수정하지 않습니다. 변경은 `src/tvr/`에서 서브클래스나 래핑으로 합니다.
- 새 기능은 CLI 플래그 뒤에 둡니다. 플래그를 끄면 베이스라인과 동일해야 하며, 이를 테스트로 증명합니다.
- 보고서 문장은 작성하지 않습니다(과제 규정). 결과는 JSON으로만 남기고 숫자를 지어내지 않습니다.
- 끝나기 전에 `PYTHONPATH=src python -m pytest -q tests/`를 통과시키고 `docs/HANDOFF.md`에 세션 로그를 남깁니다.

설정:
- `bash scripts/setup_xpool.sh`
- `pip install -r requirements-colab.txt`
- 로컬 CPU 테스트는 tiny CLIP을 monkeypatch해서 쓰므로 GPU와 인터넷 없이 실행됩니다.
