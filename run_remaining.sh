#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-logs_rerun}"

get_arg_value() {
  local flag="$1"
  shift
  local -a argv=("$@")
  local i
  for ((i = 0; i < ${#argv[@]}; i++)); do
    if [[ "${argv[$i]}" == "${flag}" ]]; then
      echo "${argv[$((i + 1))]}"
      return 0
    fi
  done
  return 1
}

has_flag() {
  local flag="$1"
  shift
  local -a argv=("$@")
  local a
  for a in "${argv[@]}"; do
    [[ "$a" == "$flag" ]] && return 0
  done
  return 1
}

compute_log_path() {
  local -a argv=("$@")

  local dataset seed gt gd label_source em_match em_select small_dim target log_file base_name suffix log_dir

  dataset="$(get_arg_value --dataset "${argv[@]}" || echo "mnist")"
  seed="$(get_arg_value --seed "${argv[@]}" || echo "0")"
  gt="$(get_arg_value --gt-domains "${argv[@]}" || echo "0")"
  gd="$(get_arg_value --generated-domains "${argv[@]}" || echo "3")"
  label_source="$(get_arg_value --label-source "${argv[@]}" || echo "pseudo")"
  em_match="$(get_arg_value --em-match "${argv[@]}" || echo "prototypes")"
  em_select="$(get_arg_value --em-select "${argv[@]}" || echo "bic")"
  small_dim="$(get_arg_value --small-dim "${argv[@]}" || echo "2048")"
  if has_flag --em-ensemble "${argv[@]}"; then
    suffix="_em-ensemble"
  else
    suffix=""
  fi

  if [[ "$dataset" == "mnist" ]]; then
    target="$(get_arg_value --rotation-angle "${argv[@]}" || echo "")"
    [[ -n "$target" ]] || { echo "ERROR: missing --rotation-angle for MNIST" >&2; return 2; }
    log_dir="${LOG_ROOT}/${dataset}/s${seed}/target${target}"
  else
    log_dir="${LOG_ROOT}/${dataset}/s${seed}"
  fi

  log_file="$(get_arg_value --log-file "${argv[@]}" || echo "")"
  if [[ -n "$log_file" ]]; then
    base_name="$log_file"
  else
    base_name="test_acc_dim${small_dim}_int${gt}_gen${gd}_${label_source}_${em_match}_${em_select}${suffix}"
  fi

  echo "${log_dir}/${base_name}.txt"
}

input="${1:--}"
if [[ "$input" != "-" && ! -f "$input" ]]; then
  echo "Usage: bash run_remaining.sh [commands_file|-]" >&2
  exit 2
fi

pending_pre_cmds=()
while IFS= read -r line; do
  [[ -z "${line//[[:space:]]/}" ]] && continue
  if [[ "$line" == rm\ -rf* ]]; then
    pending_pre_cmds+=("$line")
    continue
  fi
  [[ "$line" == python\ experiment_refrac.py* ]] || continue
  read -r -a argv <<< "$line"
  log_path="$(compute_log_path "${argv[@]}")"
  if [[ -f "$log_path" ]]; then
    echo "Skip (log exists): $log_path"
    pending_pre_cmds=()
    continue
  fi
  if (( ${#pending_pre_cmds[@]} > 0 )); then
    for pre_cmd in "${pending_pre_cmds[@]}"; do
      echo "Pre-run: $pre_cmd"
      eval "$pre_cmd"
    done
    pending_pre_cmds=()
  fi
  echo "Running: $line"
  eval "$line"
done < "$input"
