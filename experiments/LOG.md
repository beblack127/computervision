# 실험 기록 (추가만 함)

규칙: 모든 실행(실패 포함)을 한 줄씩 남깁니다. 수치는 해당 run의 JSON에서만 복사합니다.
- `E###` = full run(end-to-end), `P###` = 프록시(고정 특징), `B###` = 벤치마크/점검
- 보고 지표: t2v R@1 / R@5 / R@10 / MdR / MnR. **best**는 공식 선택 규칙(best-on-test), **last**는 마지막 체크포인트
- 결과 파일: `experiments/results/<run_name>/`

| ID | 날짜 | 실행자 | run 이름 | 커밋 | 설정(공식과 다른 점만) | seed | R@1 best | R@5 | R@10 | MdR | MnR | R@1 last | GPU·시간 | 메모 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B001 | 11/12 | 사용자 | official_9k_eval | b90b29c | 공식 체크포인트 재평가 | – | 46.9 | 72.8 | 82.2 | 2.0 | 14.3 | – | A100-SXM4-40GB · 평가 43.6s | go/no-go. 논문 값과 R@1/5/10, MdR, MnR 모두 일치. v2t R@1 44.4 |
| B002 | 11/12 | 사용자 | bench_9k | b90b29c | 200 step 속도 측정 | 24 | – | – | – | – | – | – | A100-SXM4-40GB · 0.704 s/step, 평가 1회 44.0s, 추정 6.1h/run, peak 17.05GB | go/no-go. data_wait 0.02% → GPU 연산이 병목 |
| E001 | 11/12–13 | 사용자 | xpool_9k_s24 | b90b29c | 없음(공식) | 24 | 46.7 | 71.9 | 82.0 | 2.0 | 13.8 | 46.7 | A100 · 세션 3개, 합 ≥7.27h(재학습 포함) · peak 17.56GB(마지막 세션, 80GB) | 재현. best=last(최고 window가 마지막 step 28125). 순서 probe: reversed·shuffled 모두 46.7(동일), middle 29.9. 세션 1은 Drive 쓰기 오류로 중단 → epoch 5 재학습, 재개 결과 동일. 상세 `xpool_9k_s24/sessions_reconstructed.json` |
| E002 | | 사용자 | xpool_9k_s0 | | 없음(공식) | 0 | | | | | | | | 재현 |
| E003 | | 사용자 | xpool_9k_s1 | | 없음(공식) | 1 | | | | | | | | 재현 |
