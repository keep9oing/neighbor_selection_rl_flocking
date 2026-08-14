# PLAN.md — acs-confirm

## Phase 0 — 스켈레톤 + 사전등록  [x]
- [x] 스켈레톤 + .gitignore + 러너 사본(STUDY 상수만 변경, diff 검증).
- [x] PROBLEM.md 등록 초안 (2026-08-11 21:42 UTC, 신선-시드 데이터 전).
- [x] 사용자 비준 확인: **즉시 비준** (2026-08-11 21:52 UTC) + Phase 5 실행 승인.

## Phase 1 — 게이트  [x]
- [x] G1: k12@L250 s1000–1007 vs r2 npz bit-exact (equal_nan=True).
- [x] G2: π_E@L250 s1000–1003 vs r3 C1_i80_L250_s500: t_fire 동일, dJ=0.
- [x] G3: 신선-시드 스모크 k12@L250 s1500–1503.
- [x] G1 게이트 npz는 data/gates/로 이동(신선-시드 요약 오염 방지).

## Phase 2 — 롤아웃 (워커 합계 ≤56)  [x]
- [x] Wave 1 (N40 병목 우선): π_E@N40L354 20w + π_R@N40L354 20w +
      체인 16w: k39@N40 → k9@N10 → N20 k-NN 11레인(s1500–1999) →
      π_E@N10L177 → π_R@N10L177.
- [x] Wave 2 (N40 정책 완료 후, 40w 해방): π_E/π_R @ L125 → L250 → L500
      (쌍으로 20w×2, 3배치).

## Phase 3 — 판정 + 통계 (비준 후에만)  [x]
- [x] src/confirm_judge.py: P1–P5 기계 판정 (실패수/Wilson/exact McNemar/
      CVaR10/쌍대 중앙값·Wilcoxon·부호검정), r2 src 패턴 참조.
- [x] 새니티: 아카이브 재현(r3 k12@L125 20/500 fail 등) 후 신선 데이터 판정.

## Phase 4 — 보고  [x]
- [x] REPORT_KO.md: 등록 항목별 PASS/FAIL + 실측, 논문용 표 2개
      (확증 신뢰성 표, N축 완성 표), 한계 명시.
- [x] 미팅 브리프 아티팩트 §next 갱신 (기존 url 재사용, 새로 만들지 않음).
- [x] RUNLOG 벽시계 mtime 대조, HANDOFF_NOTES 마무리.

## 옵션 Phase 5 (사용자 승인 시에만) — nearest-projection ablation
- [x] eval_c2_r3 ForensicsWrapper 패턴 래퍼(~40줄) + π_E@L250(+L125) 신선 500.

## 설계 결정 (재론 금지)

| 결정 | 근거 |
|---|---|
| 신선 시드 1500–1999 / N축 1000–1499 | ①은 오염-무결(어떤 암도 본 적 없음); ③은 기존 k-참조와 시드-쌍대 |
| 등록이 G3 스모크보다 선행 | 타임스탬프로 forking-paths 차단 |
| N40 레인 최우선 착수 | 벽시계 병목(에피소드당 비용 최대) |
| 오라클 k는 r3 A3 사후 최소실패 k로 고정(k11/k13/k10) | "튜닝된 k조차 결함" 클레임의 최강 대조군 |
| N40 McNemar는 k24·k28 둘 다 p<0.05 요구 | 동률 최선의 보수적 해석, 데이터 전 고정 |
| 워커 합계 ≤56 | 세션 예산 64스레드 − 메인/판정 여유 ~8 |
