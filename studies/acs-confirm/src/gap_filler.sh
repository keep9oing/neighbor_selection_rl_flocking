#!/bin/bash
# Waits for chain16's last lane (piR_N10L177) to finish, then fills the freed
# 16 workers with the user-approved Phase 5 ablation lanes. Total concurrent
# rollout workers stay <= 56 (N40/wave2 = 40w elsewhere + 16w here).
set -u
cd /workspace/studies/acs-confirm
C1=/workspace/test_results/c2C1_ft40_lmix_260808/manual/checkpoint_000080

until [ -f data/eval/piR_N10L177_summary.csv ] && ! pgrep -f "piR_N10L177" > /dev/null; do
  sleep 120
done
echo "[$(date -u '+%F %T')] chain16 finished -> ablation lanes on 16w"
python src/eval_project_nearest.py --ckpt "$C1" --label ablE_L250 --seeds 1500-1999 --bound 250 --workers 16 >> logs/ablE_L250.log 2>&1
echo "[$(date -u '+%F %T')] ablE_L250 done (exit $?)"
python src/eval_project_nearest.py --ckpt "$C1" --label ablE_L125 --seeds 1500-1999 --bound 125 --workers 16 >> logs/ablE_L125.log 2>&1
echo "[$(date -u '+%F %T')] ablE_L125 done (exit $?)"
echo "[$(date -u '+%F %T')] GAP-FILLER DONE"
