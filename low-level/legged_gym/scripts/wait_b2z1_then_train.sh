#!/usr/bin/env bash
# Run this from another terminal or tmux session, not the training pane itself.
# Usage: bash wait_b2z1_then_train.sh [b2z1:0.0]
set -euo pipefail

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

target="${1:-b2z1:0.0}"
poll_seconds="${B2Z1_QUEUE_POLL_SECONDS:-5}"
[[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || die 'B2Z1_QUEUE_POLL_SECONDS must be a positive integer.'
for tool in tmux ps pgrep flock; do
    command -v "$tool" >/dev/null || die "Required command not found: $tool"
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
checkpoint="$project_dir/logs/b2z1-low/b2_z1_change_ee_pos_range/model_37000.pt"

pane_id="$(tmux display-message -p -t "$target" '#{pane_id}')" || die "Cannot find tmux target: $target"
[[ -n "$pane_id" ]] || die "Cannot find tmux target: $target"
[[ "${TMUX_PANE:-}" != "$pane_id" ]] || die 'Start this waiter in another terminal or tmux session.'
pane_pid="$(tmux display-message -p -t "$pane_id" '#{pane_pid}')"
pane_tty="$(tmux display-message -p -t "$pane_id" '#{pane_tty}')"
shell_name="$(ps -p "$pane_pid" -o comm=)" || die 'The pane shell has exited.'
shell_name="${shell_name//[[:space:]]/}"
[[ "$shell_name" == bash ]] || die "Expected a Bash pane; found: $shell_name"

# Keep only one waiter for this server and pane. flock releases on exit.
server_pid="$(tmux display-message -p -t "$pane_id" '#{pid}')"
lock_file="${TMPDIR:-/tmp}/b2z1-training-queue-${UID}-${server_pid}-${pane_id#%}.lock"
exec 9>"$lock_file"
flock -n 9 || die "A waiter is already running for $target ($pane_id)."

log "Waiting for $target ($pane_id) to finish training and return to Bash."
idle_checks=0
while true; do
    pane_state="$(tmux display-message -p -t "$pane_id" '#{pane_pid} #{pane_dead} #{pane_current_command}')" \
        || die 'The target pane disappeared; no command was sent.'
    read -r current_pid pane_dead current_command <<< "$pane_state"
    [[ "$current_pid" == "$pane_pid" && "$pane_dead" == 0 ]] \
        || die 'The target pane exited or was replaced; no command was sent.'

    groups="$(ps -p "$pane_pid" -o pgid=,tpgid=)" || die 'The pane shell has exited.'
    read -r shell_pgid foreground_pgid <<< "$groups"
    # Also wait for train.py jobs that have been moved into the background.
    training_active=0
    if pgrep -t "${pane_tty#/dev/}" -f '(^|[[:space:]])([^[:space:]]*/)?train\.py([[:space:]]|$)' >/dev/null; then
        training_active=1
    else
        status=$?
        [[ "$status" == 1 ]] || die 'Could not inspect training processes.'
    fi
    if [[ "$foreground_pgid" == "$shell_pgid" && "$current_command" == bash && "$training_active" == 0 ]]; then
        idle_checks=$((idle_checks + 1))
    else
        idle_checks=0
    fi
    # Observe an idle shell twice so training cleanup can finish first.
    if (( idle_checks >= 2 )); then
        break
    fi
    sleep "$poll_seconds"
done

[[ -s "$checkpoint" ]] || die "Checkpoint missing or empty: $checkpoint. No command was sent."

training_command=(
    python train.py
    --headless
    --exptid b2_z1_change_ee_orn_range
    --proj_name b2z1-low
    --task b2z1
    --resumeid b2_z1_change_ee_pos_range
    --checkpoint 37000
    --max_iterations 45000
    --sim_device cuda:0
    --rl_device cuda:0
    --observe_gait_commands
)
printf -v quoted_training '%q ' "${training_command[@]}"
printf -v command_line 'cd -- %q && %s' "$script_dir" "$quoted_training"

# Send only after the original Bash shell owns the terminal again. Executing
# there preserves its active Conda environment and other exported variables.
tmux send-keys -t "$pane_id" C-u
tmux send-keys -t "$pane_id" -l "$command_line"
tmux send-keys -t "$pane_id" Enter
log "Submitted the next training command to $target ($pane_id)."
log "$command_line"
