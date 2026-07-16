#!/usr/bin/env bash
set -euo pipefail
cd ~/GOAT

LOG_ROOT="${LOG_ROOT:-logs_rerun}"
PLOT_ROOT="${PLOT_ROOT:-plots_rerun}"

pick_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then echo 0; return 0; fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits | \
  awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); printf "%012d %03d %s\n", $2, 1000-$3, $1}' | sort -r | head -1 | awk '{print $3}'
}

run_exp() {
  local cli_ds="$1" seed="$2" gt="$3" gen="$4" degree="${5:-}"
  local gpu
  gpu="$(pick_gpu)"
  echo "[GPU ${gpu}] ${cli_ds} seed=${seed} gt=${gt} gen=${gen} degree=${degree}"
  if [[ "${cli_ds}" == "mnist" ]]; then
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python experiment_refrac.py --plot-root "${PLOT_ROOT}" --log-root "${LOG_ROOT}" \
      --dataset mnist --rotation-angle "${degree}" --label-source pseudo --em-match prototypes --em-select bic \
      --seed "${seed}" --gt-domains "${gt}" --generated-domains "${gen}" --num-workers 0
  else
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    python experiment_refrac.py --plot-root "${PLOT_ROOT}" --log-root "${LOG_ROOT}" \
      --dataset "${cli_ds}" --label-source pseudo --em-match prototypes --em-select bic \
      --seed "${seed}" --gt-domains "${gt}" --generated-domains "${gen}" --num-workers 0
  fi
}
rm -f "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int0_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int0_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s2/test_acc_dim54_int0_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s2/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s2/small_dim54" "color_mnist/cache0.1/s2/small_dim2048"
run_exp "color_mnist" "2" "0" "1"

rm -f "${LOG_ROOT}/color_mnist/s3/test_acc_dim54_int0_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s3/test_acc_dim54_int0_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s3/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s3/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s3/test_acc_dim54_int0_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s3/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s3/small_dim54" "color_mnist/cache0.1/s3/small_dim2048"
run_exp "color_mnist" "3" "0" "2"

rm -f "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int0_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int0_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s4/test_acc_dim54_int0_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s4/test_acc_dim2048_int0_gen2_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s4/small_dim54" "color_mnist/cache0.1/s4/small_dim2048"
run_exp "color_mnist" "4" "0" "2"

rm -f "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s0/small_dim54" "color_mnist/cache0.1/s0/small_dim2048"
run_exp "color_mnist" "0" "0" "3"

rm -f "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int0_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s4/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s4/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s4/small_dim54" "color_mnist/cache0.1/s4/small_dim2048"
run_exp "color_mnist" "4" "0" "3"

rm -f "${LOG_ROOT}/color_mnist/s3/test_acc_dim54_int1_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s3/test_acc_dim54_int1_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s3/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s3/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s3/test_acc_dim54_int1_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s3/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s3/small_dim54" "color_mnist/cache0.1/s3/small_dim2048"
run_exp "color_mnist" "3" "1" "3"

rm -f "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s4/small_dim54" "color_mnist/cache0.1/s4/small_dim2048"
run_exp "color_mnist" "4" "2" "0"

rm -f "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int2_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int2_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int2_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int2_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s2/test_acc_dim54_int2_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s2/test_acc_dim2048_int2_gen1_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s2/small_dim54" "color_mnist/cache0.1/s2/small_dim2048"
run_exp "color_mnist" "2" "2" "1"

rm -f "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen1_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s4/small_dim54" "color_mnist/cache0.1/s4/small_dim2048"
run_exp "color_mnist" "4" "2" "1"

rm -f "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int2_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int2_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s0/test_acc_dim54_int2_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s0/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s0/small_dim54" "color_mnist/cache0.1/s0/small_dim2048"
run_exp "color_mnist" "0" "2" "2"

rm -f "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s4/test_acc_dim54_int2_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s4/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s4/small_dim54" "color_mnist/cache0.1/s4/small_dim2048"
run_exp "color_mnist" "4" "2" "2"

rm -f "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int2_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int2_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s0/test_acc_dim54_int2_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s0/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s0/small_dim54" "color_mnist/cache0.1/s0/small_dim2048"
run_exp "color_mnist" "0" "2" "3"

rm -f "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim54_int3_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s4/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s4/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s4/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s4/small_dim54" "color_mnist/cache0.1/s4/small_dim2048"
run_exp "color_mnist" "4" "3" "2"

rm -f "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "color_mnist/cache0.1/s1/small_dim54" "color_mnist/cache0.1/s1/small_dim2048"
run_exp "color_mnist" "1" "3" "3"

rm -f "${LOG_ROOT}/covtype/s0/test_acc_dim54_int2_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim54_int2_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s0/test_acc_dim54_int2_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s0/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "0" "2" "0"

rm -f "${LOG_ROOT}/covtype/s1/test_acc_dim54_int2_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim54_int2_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s1/test_acc_dim54_int2_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s1/test_acc_dim2048_int2_gen0_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "1" "2" "0"

rm -f "${LOG_ROOT}/covtype/s2/test_acc_dim54_int3_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim54_int3_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s2/test_acc_dim54_int3_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s2/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "2" "3" "0"

rm -f "${LOG_ROOT}/covtype/s3/test_acc_dim54_int3_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s3/test_acc_dim54_int3_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s3/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s3/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s3/test_acc_dim54_int3_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s3/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "3" "3" "0"

rm -f "${LOG_ROOT}/covtype/s4/test_acc_dim54_int3_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s4/test_acc_dim54_int3_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s4/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s4/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s4/test_acc_dim54_int3_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s4/test_acc_dim2048_int3_gen0_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "4" "3" "0"

rm -f "${LOG_ROOT}/covtype/s0/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s0/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s0/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "0" "3" "3"

rm -f "${LOG_ROOT}/covtype/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "1" "3" "3"

rm -f "${LOG_ROOT}/covtype/s2/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s2/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s2/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "covtype/cache0.1/small_dim54" "covtype/cache0.1/small_dim2048"
run_exp "covtype" "2" "3" "3"

rm -f "${LOG_ROOT}/mnist/s1/target45/test_acc_dim54_int2_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s1/target45/test_acc_dim54_int2_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s1/target45/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s1/target45/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s1/target45/test_acc_dim54_int2_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s1/target45/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "1" "2" "3" "45"

rm -f "${LOG_ROOT}/mnist/s1/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s1/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s1/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s1/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s1/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s1/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "1" "3" "1" "45"

rm -f "${LOG_ROOT}/mnist/s3/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s3/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s3/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s3/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "3" "3" "1" "45"

rm -f "${LOG_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "4" "3" "1" "45"

rm -f "${LOG_ROOT}/mnist/s3/target45/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target45/test_acc_dim54_int3_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s3/target45/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target45/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s3/target45/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s3/target45/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "3" "3" "2" "45"

rm -f "${LOG_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "4" "3" "2" "45"

rm -f "${LOG_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target45/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target45/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target45/small_dim54" "cache0.1/target45/small_dim2048"
run_exp "mnist" "4" "3" "3" "45"

rm -f "${LOG_ROOT}/mnist/s3/target60/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target60/test_acc_dim54_int3_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s3/target60/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target60/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s3/target60/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s3/target60/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target60/small_dim54" "cache0.1/target60/small_dim2048"
run_exp "mnist" "3" "3" "1" "60"

rm -f "${LOG_ROOT}/mnist/s4/target60/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target60/test_acc_dim54_int3_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target60/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target60/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target60/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target60/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target60/small_dim54" "cache0.1/target60/small_dim2048"
run_exp "mnist" "4" "3" "1" "60"

rm -f "${LOG_ROOT}/mnist/s4/target60/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target60/test_acc_dim54_int3_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target60/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target60/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target60/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target60/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target60/small_dim54" "cache0.1/target60/small_dim2048"
run_exp "mnist" "4" "3" "2" "60"

rm -f "${LOG_ROOT}/mnist/s4/target90/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target90/test_acc_dim54_int3_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target90/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target90/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target90/test_acc_dim54_int3_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target90/test_acc_dim2048_int3_gen1_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target90/small_dim54" "cache0.1/target90/small_dim2048"
run_exp "mnist" "4" "3" "1" "90"

rm -f "${LOG_ROOT}/mnist/s3/target90/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target90/test_acc_dim54_int3_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s3/target90/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target90/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s3/target90/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s3/target90/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target90/small_dim54" "cache0.1/target90/small_dim2048"
run_exp "mnist" "3" "3" "2" "90"

rm -f "${LOG_ROOT}/mnist/s1/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s1/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s1/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s1/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s1/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s1/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target90/small_dim54" "cache0.1/target90/small_dim2048"
run_exp "mnist" "1" "3" "3" "90"

rm -f "${LOG_ROOT}/mnist/s3/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s3/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s3/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s3/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s3/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target90/small_dim54" "cache0.1/target90/small_dim2048"
run_exp "mnist" "3" "3" "3" "90"

rm -f "${LOG_ROOT}/mnist/s4/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/mnist/s4/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/mnist/s4/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/mnist/s4/target90/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/mnist/s4/target90/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
rm -rf "cache0.1/target90/small_dim54" "cache0.1/target90/small_dim2048"
run_exp "mnist" "4" "3" "3" "90"

python collect_result.py --log-bases logs,logs_rerun
