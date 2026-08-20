#!/bin/bash
# Phase 2 orchestrator — acs-confirm. Keeps total rollout workers <= 56.
# Wave 1: N40 policy lanes (bottleneck) 20w+20w + 16w chain
#         (k39@N40 -> k9@N10 -> N20 knn x11 lanes -> N10 policy lanes).
# Wave 2: after N40 policy lanes exit, N20 fresh policy pairs 20w+20w per L.
set -u
cd /workspace/studies/acs-confirm
C1=/workspace/test_results/c2C1_ft40_lmix_260808/manual/checkpoint_000080
R1="/workspace/test_results/c2R1_lmix_legacy_260807/GradLoggingPPO_neighbor_selection_flocking_env_cc4e6_00000_0_2026-08-07_01-24-56/checkpoint_000110"
EV="python src/eval_c2_r3.py"
KN="python src/run_knn_refs3.py"

echo "[$(date -u '+%F %T')] WAVE1 launch: piE@N40 20w, piR@N40 20w, chain 16w"

$EV --ckpt "$C1" --label piE_N40L354 --seeds 1000-1499 --bound 354 --n-agents 40 --workers 20 >> logs/piE_N40L354.log 2>&1 &
PE40=$!
$EV --ckpt "$R1" --label piR_N40L354 --seeds 1000-1499 --bound 354 --n-agents 40 --workers 20 >> logs/piR_N40L354.log 2>&1 &
PR40=$!

(
  $KN --k 39 --L 354 --n-agents 40 --seeds 1000-1499 --workers 16 && \
  $KN --k 9 --L 177 --n-agents 10 --seeds 1000-1499 --workers 16 && \
  $KN --k 11,12,13,19 --L 125 --seeds 1500-1999 --workers 16 && \
  $KN --k 12,13,19 --L 250 --seeds 1500-1999 --workers 16 && \
  $KN --k 10,12,13,19 --L 500 --seeds 1500-1999 --workers 16 && \
  echo "[$(date -u '+%F %T')] knn lanes done; N10 policy lanes start" && \
  $EV --ckpt "$C1" --label piE_N10L177 --seeds 1000-1499 --bound 177 --n-agents 10 --workers 16 && \
  $EV --ckpt "$R1" --label piR_N10L177 --seeds 1000-1499 --bound 177 --n-agents 10 --workers 16
) >> logs/chain16.log 2>&1 &
PCHAIN=$!

wait "$PE40"; E40=$?
wait "$PR40"; R40=$?
echo "[$(date -u '+%F %T')] WAVE1 N40 policy lanes done (exit $E40/$R40); WAVE2 pairs start"

for L in 125 250 500; do
  $EV --ckpt "$C1" --label "piE_L$L" --seeds 1500-1999 --bound "$L" --workers 20 >> "logs/piE_L$L.log" 2>&1 &
  A=$!
  $EV --ckpt "$R1" --label "piR_L$L" --seeds 1500-1999 --bound "$L" --workers 20 >> "logs/piR_L$L.log" 2>&1 &
  B=$!
  wait "$A"; EA=$?
  wait "$B"; EB=$?
  echo "[$(date -u '+%F %T')] WAVE2 L$L pair done (exit $EA/$EB)"
done

wait "$PCHAIN"; EC=$?
echo "[$(date -u '+%F %T')] chain16 done (exit $EC)"
echo "[$(date -u '+%F %T')] PHASE2 ALL DONE"
