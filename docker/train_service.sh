#!/usr/bin/env bash
set -Eeuo pipefail

training_results_dir="${TRAINING_RESULTS_DIR:-/workspace/test_results}"
workflow_run_id="${WORKFLOW_RUN_ID:-dynamic-k-nn}"
completion_marker="${training_results_dir}/.training_complete"

mkdir -p "${training_results_dir}"

if [ -f "${completion_marker}" ]; then
    echo "[training-service] run=${workflow_run_id} already completed; staying idle"
    exec sleep infinity
fi

echo "[training-service] run=${workflow_run_id} results=${training_results_dir}"
set +e
python -u train.py
training_exit_code=$?
set -e

if [ "${training_exit_code}" -ne 0 ]; then
    echo "[training-service] training exited with code ${training_exit_code}; Docker will restart the container" >&2
    exit "${training_exit_code}"
fi

temporary_marker="${completion_marker}.tmp"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${temporary_marker}"
mv "${temporary_marker}" "${completion_marker}"
echo "[training-service] training complete; staying idle"
exec sleep infinity
