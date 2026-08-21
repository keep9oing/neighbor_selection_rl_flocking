#!/bin/bash
# Evaluate the latest continuous action checkpoint against FC-ACS
# Usage: CUDA_VISIBLE_DEVICES=1 bash run_eval.sh [checkpoint_path]

CHECKPOINT=${1:-$(find /workspace/test_results/continuous_sf15_260526 -name "checkpoint_*" -type d | sort | tail -1)}

if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found"
    exit 1
fi

echo "Evaluating checkpoint: $CHECKPOINT"
python evaluate_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --num_episodes 100 \
    --num_agents 20 \
    --max_steps 1000 \
    --seed 42
