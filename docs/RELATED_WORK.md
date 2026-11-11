# 선행연구 지도: X-Pool 주변

> 논문 이해용 지도입니다. 수치는 아래 출처에서 확인한 것만 적었고, 확인 못 한 칸은 `–`입니다.
> 보고서 Related paper review는 각 논문을 직접 읽고 본인 말로 쓰세요.

## 1. 계보

```
CLIP (ICML'21)
 └─ CLIP4Clip (Neurocomputing'22): 프레임 평균/시퀀스 풀링으로 CLIP을 영상 검색에 이식
     └─ X-Pool (CVPR'22) ★ 베이스라인: 텍스트가 프레임에 attention하는 "텍스트-조건부 풀링"
         ├─ 텍스트 쪽을 풍부하게: T-MASS (CVPR'24, X-Pool 위에 구현), UATVR (ICCV'23)
         ├─ 정렬 단위를 세분화: X-CLIP (ACM MM'22), UCoFiA (ICCV'23), TS2-Net (ECCV'22)
         ├─ 보조 정보 추가: Cap4Video (CVPR'23, 자동 캡션), TEFAL (ICCV'23, 오디오)
         │                 Cap4Bridge (IEEE Access'26, 교수님 연구실, 캡션 기반)
         ├─ 생성 모델로 재정의: DiffusionRet (ICCV'23)
         └─ 테스트 시 후처리: QB-Norm (CVPR'22), Dual-Softmax(DSL). 테스트 query 묶음을 사용하므로 공정 비교에서 주의
```

## 2. MSR-VTT 1k-A, text→video, CLIP ViT-B/32

| 방법 | 학회 | 핵심 아이디어(한 줄) | R@1 | 출처 |
|---|---|---|---|---|
| X-Pool | CVPR 2022 | 텍스트→프레임 cross-attention 풀링 | 46.9 | X-Pool 논문/README, T-MASS Table 1 |
| X-CLIP | ACM MM 2022 | 영상-문장, 프레임-단어 등 다중 수준 유사도 | 46.1 | UCoFiA supp. Table 13 |
| UATVR | ICCV 2023 | 불확실성을 고려한 분포 매칭 | 47.5 | T-MASS Table 1 |
| DiffusionRet | ICCV 2023 | 검색을 결합분포 생성 문제로 봄 | 49.0 | T-MASS Table 1 |
| TEFAL | ICCV 2023 | 오디오 + 텍스트-조건부 특징 정렬 | 49.4 | T-MASS Table 1 |
| UCoFiA | ICCV 2023 | 거친-세밀 정렬 통합, Sinkhorn 정규화(hubness 완화) | 49.4 | UCoFiA 논문 |
| CLIP-ViP | ICLR 2023 | 영상-언어 후속 사전학습 | 50.1 | T-MASS Table 1 |
| T-MASS | CVPR 2024 | 텍스트를 점이 아닌 확률적 "질량"으로 모델링 | 50.2 | T-MASS Table 1 |
| Cap4Video | CVPR 2023 | zero-shot 캡션으로 증강, 상호작용, 점수 융합 | – (B/32 수치 미확인) | Cap4Video 논문 |
| Cap4Bridge | IEEE Access 2026 | 캡션 기반 문맥화 + 확률적 증강 (교수님 연구실) | – | 김동진 교수 홈페이지 |

참고: X-Pool ViT-B/16은 48.2, T-MASS ViT-B/16은 52.7입니다(T-MASS Table 1).

## 3. 우리 프로젝트와의 관계 (아이디어 겹침 점검용)

| 한계 축 | 이미 다룬 연구 | 메모 |
|---|---|---|
| 텍스트 임베딩이 점 하나 | T-MASS, UATVR | 확률적 텍스트 방향은 이미 X-Pool 위에서 검증됨. **그대로 하면 독창성이 약함** |
| 정렬 단위가 문장-프레임뿐 | X-CLIP, UCoFiA, TS2-Net | 단어/패치 수준은 계산량이 큼(UCoFiA는 GPU 4장) |
| hubness | QB-Norm, DSL, UCoFiA(Sinkhorn) | 대부분 **테스트 시 후처리**. 학습 단계에서 다루는 방향은 별도로 점검 필요 |
| 캡션 보조 정보 | Cap4Video, **Cap4Bridge(교수님 연구실)** | **피하기**. 겹치면 교수님이 바로 알아봄 |
| 시간 순서 | (점검 필요) | X-Pool 풀링은 프레임 순서에 불변(order-invariant)입니다. 우리 probe 실험으로 먼저 확인 |

## 출처
- T-MASS: https://arxiv.org/pdf/2403.17998 (Table 1), supp: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Wang_Text_Is_MASS_CVPR_2024_supplemental.pdf
- UCoFiA: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Unified_Coarse-to-Fine_Alignment_for_Video-Text_Retrieval_ICCV_2023_paper.html, supp: https://openaccess.thecvf.com/content/ICCV2023/supplemental/Wang_Unified_Coarse-to-Fine_Alignment_ICCV_2023_supplemental.pdf
- Cap4Video: https://arxiv.org/abs/2301.00184
- X-Pool: https://arxiv.org/abs/2203.15086, 코드 https://github.com/layer6ai-labs/xpool
- 교수님 연구실 논문 목록: https://sites.google.com/site/djkimcv
