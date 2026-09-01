#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${B1Z1_PYTHON:-/opt/conda/envs/b1z1/bin/python}"
gpu_id="${GPU_ID:-0}"
task="${TASK:-b2_z1}"
num_envs="${NUM_ENVS:-6144}"
max_iterations="${MAX_ITERATIONS:-45000}"
terrain_rows="${TERRAIN_ROWS:-10}"
terrain_cols="${TERRAIN_COLS:-20}"
project_name="${PROJECT_NAME:-b2z1-low}"
run_name="${RUN_NAME:-${task}_$(date -u +%Y%m%d_%H%M%S)}"
log_dir="${LOG_DIR:-${repo_dir}}"
log_file="${log_dir}/${run_name}.log"
pid_file="${log_dir}/${run_name}.pid"

if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable not found: ${python_bin}" >&2
    exit 1
fi

train_script="${repo_dir}/low-level/legged_gym/scripts/train.py"
if [[ ! -f "${train_script}" ]]; then
    echo "Training script not found: ${train_script}" >&2
    exit 1
fi

if [[ -f "${pid_file}" ]]; then
    previous_pid="$(<"${pid_file}")"
    if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Run ${run_name} is already active with PID ${previous_pid}" >&2
        exit 1
    fi
fi

mkdir -p "${log_dir}" "${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"

env_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/low-level:${repo_dir}/third_party/rsl_rl:${repo_dir}/third_party/isaacgym/python"
export LD_LIBRARY_PATH="${env_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

cd "${repo_dir}/low-level/legged_gym/scripts"

nohup setsid "${python_bin}" -u train.py \
    --headless \
    --task "${task}" \
    --exptid "${run_name}" \
    --proj_name "${project_name}" \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --pipeline gpu \
    --num_envs "${num_envs}" \
    --max_iterations "${max_iterations}" \
    --rows "${terrain_rows}" \
    --cols "${terrain_cols}" \
    >"${log_file}" 2>&1 < /dev/null &

train_pid=$!
echo "${train_pid}" > "${pid_file}"

sleep 2
if ! kill -0 "${train_pid}" 2>/dev/null; then
    echo "Training exited during startup. Last log lines:" >&2
    tail -n 40 "${log_file}" >&2 || true
    exit 1
fi

echo "B2-Z1 training started"
echo "PID: ${train_pid}"
echo "GPU: ${gpu_id}"
echo "Environments: ${num_envs}"
echo "Iterations: ${max_iterations}"
echo "Log: ${log_file}"
echo "PID file: ${pid_file}"
