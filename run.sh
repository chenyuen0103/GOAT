#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-logs_rerun}"
PLOT_ROOT="${PLOT_ROOT:-plots_rerun}"

angles=(45 90 30 60)
label_sources=(pseudo)
em_matches=(prototypes pseudo)
seeds=(0 1 2)
gt_domains=(0 1 2 3)
generated_domains=(1)
small_dim=2048
em_select="bic"

# If 1, enable EM ensembling and include its suffix in log filenames.
EM_ENSEMBLE="${EM_ENSEMBLE:-1}"
em_ensemble_suffix=""
em_ensemble_flag=()
if [[ "${EM_ENSEMBLE}" == "1" ]]; then
  em_ensemble_suffix="_em-ensemble"
  em_ensemble_flag=(--em-ensemble)
fi

# If 1, print what would run without executing Python.
DRY_RUN="${DRY_RUN:-0}"

# If 1, rerun gen=0 experiments even if logs already exist (old results were saved differently).
RERUN_GEN0="${RERUN_GEN0:-1}"

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

for a in "${angles[@]}"; do
  for ls in "${label_sources[@]}"; do
    for m in "${em_matches[@]}"; do
      for s in "${seeds[@]}"; do
        for gt in "${gt_domains[@]}"; do
          for gd in "${generated_domains[@]}"; do
            # When generated_domains=0, the Python code now only runs the GOAT baseline and
            # em_match is irrelevant. Run exactly one canonical config to avoid duplicates.
            if [[ "$gd" == "0" && "$m" != "prototypes" ]]; then
              continue
            fi

            log_base="test_acc_dim${small_dim}_int${gt}_gen${gd}_${ls}_${m}_${em_select}${em_ensemble_suffix}.txt"
            log_dir="${LOG_ROOT}/mnist/s${s}/target${a}"
            log_path="${log_dir}/${log_base}"
            if [[ -f "$log_path" ]]; then
              marker="seed${s}with${gt}gt${gd}generated,"
              if is_complete_log "$log_path" "$marker"; then
                if [[ "$gd" == "0" && "$RERUN_GEN0" == "1" ]]; then
                  ts="$(date +%Y%m%d_%H%M%S)"
                  echo "Rerun gen=0 (forced): $log_path"
                  backup_run_artifacts "$log_path" "$ts"
                else
                  echo "Skip (already ran): $log_path"
                  continue
                fi
              else
                ts="$(date +%Y%m%d_%H%M%S)"
                echo "Rerun (incomplete log): $log_path"
                backup_run_artifacts "$log_path" "$ts"
              fi

              if [[ "$gd" == "0" && "$RERUN_GEN0" == "1" ]]; then
                :
              fi
            fi
            echo "Running: angle=$a, label_source=$ls, em_match=$m, seed=$s, gt_domains=$gt, generated_domains=$gd"
            if [[ "${DRY_RUN}" == "1" ]]; then
              echo "DRY_RUN=1: python experiment_refrac.py --plot-root \"$PLOT_ROOT\" --log-root \"$LOG_ROOT\" --rotation-angle \"$a\" --label-source \"$ls\" --em-match \"$m\" --gt-domains \"$gt\" --generated-domains \"$gd\" ${em_ensemble_flag[*]} --seed \"$s\""
            else
              python experiment_refrac.py --plot-root "$PLOT_ROOT" --log-root "$LOG_ROOT" --rotation-angle "$a" --label-source "$ls" --em-match "$m" --gt-domains "$gt" --generated-domains "$gd" "${em_ensemble_flag[@]}" --seed "$s"
            fi
          done
        done
      done
    done
  done
done
