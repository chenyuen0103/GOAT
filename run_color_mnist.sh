#!/usr/bin/env bash
set -euo pipefail

label_sources=(pseudo em)
em_matches=(prototypes pseudo)
seeds=(0 1 2)
gt_domains=(0 1 2 3)
generated_domains=(0)
small_dim=2048
em_select="bic"
em_ensemble_suffix=""

# If 1, rerun gen=0 experiments even if logs already exist (old results were saved incorrectly).
RERUN_GEN0="${RERUN_GEN0:-1}"

# If 1, enable EM ensembling and include its suffix in log filenames.
EM_ENSEMBLE="${EM_ENSEMBLE:-1}"
em_ensemble_flag=()
if [[ "${EM_ENSEMBLE}" == "1" ]]; then
  em_ensemble_suffix="_em-ensemble"
  em_ensemble_flag=(--em-ensemble)
fi

# If 1, print what would run without executing Python.
DRY_RUN="${DRY_RUN:-0}"

is_complete_log() {
  local path="$1"
  local marker="$2"
  [[ -f "$path" ]] || return 1
  if command -v rg >/dev/null 2>&1; then
    rg -q --fixed-strings "$marker" "$path"
  else
    grep -Fq -- "$marker" "$path"
  fi
}

is_complete_any_log() {
  local log_path="$1"
  local marker="$2"

  if is_complete_log "$log_path" "$marker"; then
    return 0
  fi

  # If the "current" log was backed up (e.g., after an interrupt), treat any
  # completed backup as sufficient to skip rerunning this config.
  local bak
  shopt -s nullglob
  for bak in "${log_path}.bak."*; do
    if is_complete_log "$bak" "$marker"; then
      return 0
    fi
  done
  return 1
}

backup_run_artifacts() {
  local log_path="$1"
  local ts="$2"
  echo "Backing up existing logs -> ${log_path}.bak.${ts}"
  mv -f "$log_path" "${log_path}.bak.${ts}"
  local curves_path="${log_path%.txt}_curves.jsonl"
  if [[ -f "$curves_path" ]]; then
    mv -f "$curves_path" "${curves_path}.bak.${ts}"
  fi
}

# GPU selection
# - Set GPU_ID to a specific physical GPU index (e.g., 3) to pin runs.
# - Otherwise, GPU_LIST controls which GPUs are eligible (comma-separated indices).
# - AUTO_PICK_GPU=1 picks the eligible GPU with the most free memory (ties -> lower util).
AUTO_PICK_GPU="${AUTO_PICK_GPU:-1}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"

pick_gpu() {
  if [[ -n "${GPU_ID:-}" ]]; then
    echo "${GPU_ID}"
    return 0
  fi
  if [[ "${AUTO_PICK_GPU}" != "1" ]]; then
    echo "0"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "0"
    return 0
  fi

  local -a allowed
  IFS=',' read -r -a allowed <<< "${GPU_LIST}"

  local best_idx="" best_free="-1" best_util="101"
  # index, free(MiB), util(%)
  while IFS=',' read -r idx free util; do
    idx="${idx//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    [[ -z "$idx" || -z "$free" || -z "$util" ]] && continue

    local ok="0"
    for a in "${allowed[@]}"; do
      a="${a//[[:space:]]/}"
      if [[ "$idx" == "$a" ]]; then ok="1"; break; fi
    done
    [[ "$ok" != "1" ]] && continue

    # Primary: max free; Secondary: min util
    if [[ "$free" -gt "$best_free" || ( "$free" -eq "$best_free" && "$util" -lt "$best_util" ) ]]; then
      best_idx="$idx"
      best_free="$free"
      best_util="$util"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

  if [[ -z "$best_idx" ]]; then
    echo "0"
  else
    echo "$best_idx"
  fi
}

for ls in "${label_sources[@]}"; do
  for m in "${em_matches[@]}"; do
    for s in "${seeds[@]}"; do
      for gt in "${gt_domains[@]}"; do
        for gd in "${generated_domains[@]}"; do
          # When generated_domains=0, the Python code now only runs the GOAT baseline and
          # label_source / em_match are irrelevant. Run exactly one canonical config to
          # avoid duplicate work and keep filenames stable for table scripts.
          if [[ "$gd" == "0" && ( "$ls" != "pseudo" || "$m" != "prototypes" ) ]]; then
            continue
          fi

          log_file="test_acc_dim${small_dim}_int${gt}_gen${gd}_${ls}_${m}_${em_select}${em_ensemble_suffix}"
          log_path="logs/color_mnist/s${s}/${log_file}.txt"

          marker="seed${s}with${gt}gt${gd}generated,"
          if is_complete_any_log "$log_path" "$marker"; then
            if [[ "$gd" == "0" && "$RERUN_GEN0" == "1" ]]; then
              ts="$(date +%Y%m%d_%H%M%S)"
              echo "Rerun gen=0 (forced): $log_path"
              [[ -f "$log_path" ]] && backup_run_artifacts "$log_path" "$ts"
            else
              echo "Skip (already ran): $log_path"
              continue
            fi
          elif [[ -f "$log_path" ]]; then
            ts="$(date +%Y%m%d_%H%M%S)"
            echo "Rerun (incomplete log): $log_path"
            backup_run_artifacts "$log_path" "$ts"
          fi

          echo "Running: dataset=color_mnist, label_source=$ls, em_match=$m, seed=$s, gt_domains=$gt, generated_domains=$gd"
          gpu="$(pick_gpu)"
          echo "Using GPU ${gpu} (set GPU_ID to override; GPU_LIST=${GPU_LIST})"
          if [[ "${DRY_RUN}" == "1" ]]; then
            echo "DRY_RUN=1: CUDA_VISIBLE_DEVICES=\"${gpu}\" PYTORCH_CUDA_ALLOC_CONF=\"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}\" python experiment_refrac.py --dataset color_mnist --label-source \"$ls\" --em-match \"$m\" --seed \"$s\" --gt-domains \"$gt\" --generated-domains \"$gd\" --log-file \"$log_file\" ${em_ensemble_flag[*]}"
          else
            CUDA_VISIBLE_DEVICES="${gpu}" \
              PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
              python experiment_refrac.py --dataset color_mnist --label-source "$ls" --em-match "$m" --seed "$s" --gt-domains "$gt" --generated-domains "$gd" --log-file "$log_file" "${em_ensemble_flag[@]}"
          fi
        done
      done
    done
  done
done
