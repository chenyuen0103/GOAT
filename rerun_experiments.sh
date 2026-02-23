#!/usr/bin/env bash
set -euo pipefail
cd ~/GOAT

LOG_ROOT="${LOG_ROOT:-logs_rerun}"
PLOT_ROOT="${PLOT_ROOT:-plots_rerun}"
FORCE_RERUN="${FORCE_RERUN:-0}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
  awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); score=$2 + 200*$3; print score, $1}' |
  sort -n | head -1 | awk '{print $2}'
}

run_exp() {
  local ds="$1" seed="$2" gt="$3" gen="$4"
  local gpu
  gpu="$(pick_gpu)"
  echo "[GPU ${gpu}] ${ds} seed=${seed} gt=${gt} gen=${gen}"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python experiment_refrac.py --plot-root "${PLOT_ROOT}" --log-root "${LOG_ROOT}" --dataset "${ds}" --label-source pseudo --em-match prototypes --seed "${seed}" --gt-domains "${gt}" --generated-domains "${gen}" --num-workers 0
}

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

clean_run_artifacts() {
  local ds="$1" seed="$2" gt="$3" gen="$4"
  rm -f \
    "${LOG_ROOT}/${ds}/s${seed}/test_acc_dim54_int${gt}_gen${gen}_pseudo_prototypes_bic.txt" \
    "${LOG_ROOT}/${ds}/s${seed}/test_acc_dim54_int${gt}_gen${gen}_pseudo_prototypes_bic_curves.jsonl" \
    "${LOG_ROOT}/${ds}/s${seed}/test_acc_dim2048_int${gt}_gen${gen}_pseudo_prototypes_bic.txt" \
    "${LOG_ROOT}/${ds}/s${seed}/test_acc_dim2048_int${gt}_gen${gen}_pseudo_prototypes_bic_curves.jsonl"
}

run_or_skip() {
  local ds="$1" seed="$2" gt="$3" gen="$4"
  local marker="seed${seed}with${gt}gt${gen}generated,"
  local log54="${LOG_ROOT}/${ds}/s${seed}/test_acc_dim54_int${gt}_gen${gen}_pseudo_prototypes_bic.txt"
  local log2048="${LOG_ROOT}/${ds}/s${seed}/test_acc_dim2048_int${gt}_gen${gen}_pseudo_prototypes_bic.txt"

  if [[ "${FORCE_RERUN}" != "1" ]] &&
     is_complete_log "$log54" "$marker" &&
     is_complete_log "$log2048" "$marker"; then
    echo "Skip (already complete): ${ds} seed=${seed} gt=${gt} gen=${gen}"
    return 0
  fi

  clean_run_artifacts "$ds" "$seed" "$gt" "$gen"
  run_exp "$ds" "$seed" "$gt" "$gen"
}

run_or_skip "color_mnist" "0" "0" "3"
run_or_skip "color_mnist" "0" "1" "1"
run_or_skip "color_mnist" "1" "1" "2"
run_or_skip "color_mnist" "1" "1" "3"
run_or_skip "color_mnist" "1" "2" "2"
run_or_skip "color_mnist" "1" "2" "3"
run_or_skip "color_mnist" "1" "3" "3"
run_or_skip "color_mnist" "2" "0" "1"
run_or_skip "color_mnist" "2" "3" "2"

python collect_result.py
