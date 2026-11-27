#!/usr/bin/env bash
# Run from the same activated Python environment used for run_base.py.
# ER alone is omitted because that experiment has already completed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"

CONFIG="configs/base_alllayers_config.yml"
TASKS=(ks si ks_si ks_er si_er ks_si_er)
LOG_DIR="logs/alllayers_combinations_$(date -u +%Y_%m_%d_%H_%M_%S)_$$"
mkdir -p -- "$LOG_DIR"

printf 'Config: %s\nLogs: %s/%s\n' "$CONFIG" "$SCRIPT_DIR" "$LOG_DIR"

# Run one experiment at a time on the GPU configured in the YAML.
# Stop on failure so an unsuccessful experiment is easy to identify.
for task in "${TASKS[@]}"; do
    log_file="$LOG_DIR/${task}.log"
    printf '[%s] Starting %s; log: %s\n' "$(date -u +%FT%TZ)" "$task" "$log_file"
    if python -u run_base.py --task_type "$task" --config "$CONFIG" > "$log_file" 2>&1; then
        printf '[%s] Completed %s\n' "$(date -u +%FT%TZ)" "$task"
    else
        status=$?
        printf '[%s] FAILED %s (exit %s). See %s/%s\n' "$(date -u +%FT%TZ)" "$task" "$status" "$SCRIPT_DIR" "$log_file" >&2
        exit "$status"
    fi
done

printf '[%s] All six task combinations completed.\n' "$(date -u +%FT%TZ)"
