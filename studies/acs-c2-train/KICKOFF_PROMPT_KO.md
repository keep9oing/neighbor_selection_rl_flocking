# 새 세션 시작 프롬프트 (아래 전체를 복사해서 새 세션에 붙여넣기)

---

이번 세션의 임무: **C2 수렴 기준 아래에서 이웃 선택 정책을 재학습**하는 스터디
`studies/acs-c2-train/`을 구현하고 학습·평가까지 실행하는 것.

## 먼저 읽을 것 (이 순서대로)

1. `/workspace/studies/acs-c2-train/PROBLEM.md` — 목표·배경·성공 기준·제약
2. `/workspace/studies/acs-c2-train/PLAN.md` — 구현 계획 전체 (Phase 0-5,
   설계 스펙과 파일 앵커 포함). **이 계획을 따르되, 코드 실측과 어긋나면
   계획을 맹신하지 말고 코드를 우선하고 RUNLOG에 편차를 기록할 것.**
3. `/workspace/studies/acs-c2-train/RUNLOG.md` — 지금까지 실제로 한 일
4. 필요 시 배경: `studies/acs-conv-knn/RUNLOG.md`의 2026-08-06 항목 3개
   (C2 창 스윕, 평가지표 확정, 아키텍처 포화 진단), `NOTES_env.md`

## 핵심 요약 (자세한 건 위 문서들이 우선)

- 수렴 기준 C2: φ>0.98 50스텝 유지 ∧ 전원 단일 r0-근접 성분 300스텝 유지 ∧
  σ_p 상대 p2p(300스텝) < 5%. 성공 시에만 조기 종료, 실패는 캡(1500)까지.
- 보상: 절대 앵커 제거 → 응집 −4·(1−f_largest)² + 정렬 −0.2·max(0.98−φ,0)² +
  제어비 0.1 + 성공 보너스 +10.
- 이전 승자(hardtopk10)는 로짓 ±20 포화로 정책 그래디언트가 0이었고 aux가
  만든 kNN(10) 모방이었음 — **이번엔 비포화가 생명선** (grad norm으로 검증).
- 두 변형 병렬 학습: A(bernoulli, 사전지식 없음, cuda:1) /
  B(threshold 적응 선택 + dist_aux 어닐 1.0→0.2, cuda:3).
  **B를 kNN 모방으로 조작하지 말 것** (rank-deviation 지표로 확인).
- 평가: 성공률 → t_conv·J(=회전에너지+ρ·dt·t_conv) → 품질 마진. 프런티어
  기준선: FC J=228(32/32), k=12 J=160(31/32), k=10 J=171(29/32), 구 NN
  J=166(16/16). 목표: 성공≥31/32 & J≤160.

## 제약 (전부 구속력 있음)

- GPU는 **cuda:1, cuda:3만** (CUDA_VISIBLE_DEVICES로 강제; cuda:0/2 금지),
  CPU 총 64 threads 이하.
- **git push 절대 금지** (요청받아도 금지). 커밋은 내 명시 승인 시에만,
  커밋 메시지에 AI 언급/Co-Authored-By 금지.
- repo 코드 수정 허용됨. 단 **추가적(additive)·플래그 게이팅** 방식으로:
  새 config 필드 기본값 = 기존 동작, 기존 진입점(train.py, train_hardtopk.py,
  evaluate_checkpoint.py, test_baselines.py)은 그대로 작동해야 하고, 수정 후
  회귀 확인(PLAN Phase 0-2의 검증 게이트) 필수.
- 무거운 산출물(체크포인트/npz/png/log)은 커밋 금지. 체크포인트는
  /workspace/test_results/, 스터디 산출물은 studies/acs-c2-train/ 아래
  (data/figs/logs는 .gitignore 됨).
- 스택 고정: Pydantic v1, gym 0.23.1, Ray 2.1.0, Torch 1.12.1, NumPy 1.23.4.
- 나와의 대화·보고는 한국어, 코드·내부 문서는 영어.
- PROBLEM/PLAN/RUNLOG를 항상 최신으로 유지 (세션 핸드오프 품질).

## 진행 방식

- PLAN의 Phase 0(접지+회귀 베이스라인)부터 순서대로. 각 Phase의 검증 게이트를
  통과하고 RUNLOG에 기록한 뒤 다음으로.
- Phase 3에서 학습 시작 전 nvidia-smi로 cuda:1/3 여유 확인 후 A/B 동시 런칭,
  이후 모니터링 계획(PLAN Phase 3)대로 주기 점검. 학습이 도는 동안 Phase 4
  평가 스크립트를 준비해 둘 것.
- 막히면 스스로 조사해 해결하되, PLAN의 "Design decisions already made" 표에
  있는 항목은 내 입력 없이 뒤집지 말 것. 실험 중 계획 변경이 필요하면 근거와
  함께 물어볼 것.
- 완료(또는 중간 마일스톤) 시 한국어로 결과 보고: 숫자 우선, 유의성에 정직하게.
