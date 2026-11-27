#!/usr/bin/env bash
# Usage: bash configs/task_combinations/run_gate_0p30_0p50_pairs_30ep.sh [GPU_INDEX]
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clustering_dir="$(cd "${script_dir}/../.." && pwd)"
repository_dir="$(cd "${clustering_dir}/../.." && pwd)"
venv_activate="${repository_dir}/.venv/bin/activate"
log_dir="${clustering_dir}/logs"
gpu_index="${1:-0}"

if [[ ! "${gpu_index}" =~ ^[0-9]+$ ]]; then
  echo "GPU_INDEX must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! -f "${venv_activate}" ]]; then
  echo "Virtual environment activation script not found: ${venv_activate}" >&2
  exit 1
fi
source "${venv_activate}"
python_bin="${PYTHON_BIN:-python}"
mkdir -p "${log_dir}"
cd "${clustering_dir}"

task_types=(ks_si si_er ks_er)
config="configs/task_combinations/gate_0p30_0p50_pairs_seed42_30ep.yml"
for task_type in "${task_types[@]}"; do
  run_log="${log_dir}/${task_type}_gate_0p30_0p50_seed42_30ep_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "Starting ${task_type} on GPU ${gpu_index}; console log: ${run_log}"
  if "${python_bin}" -u run.py \
      --task_type "${task_type}" \
      --config "${config}" \
      --device_index "${gpu_index}" \
      > "${run_log}" 2>&1; then
    echo "Completed ${task_type}"
  else
    status=$?
    echo "${task_type} failed with exit status ${status}; inspect ${run_log}" >&2
    exit "${status}"
  fi
done
echo "All three 0.30/0.50 gated pair runs completed."
