#!/usr/bin/env bash
set -euo pipefail
cd ~/GOAT

LOG_ROOT="${LOG_ROOT:-logs_rerun}"
PLOT_ROOT="${PLOT_ROOT:-plots_rerun}"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
  awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); score=$2 + 200*$3; print score, $1}' |
  sort -n | head -1 | awk '{print $2}'
}

run_exp() {
  local cli_ds="$1" seed="$2" gt="$3" gen="$4"
  local gpu
  gpu="$(pick_gpu)"
  echo "[GPU ${gpu}] ${cli_ds} seed=${seed} gt=${gt} gen=${gen}"
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python experiment_refrac.py --plot-root "${PLOT_ROOT}" --log-root "${LOG_ROOT}" \
    --dataset "${cli_ds}" --label-source pseudo --em-match prototypes \
    --seed "${seed}" --gt-domains "${gt}" --generated-domains "${gen}" --num-workers 0
}

rm -f "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic.png"
run_exp "color_mnist" "0" "0" "1"

rm -f "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s0/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s0/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.png"
run_exp "color_mnist" "0" "0" "3"

rm -f "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int1_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int1_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s2/test_acc_dim54_int1_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s2/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic.png"
run_exp "color_mnist" "2" "1" "1"

rm -f "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int1_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int1_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s1/test_acc_dim54_int1_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s1/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic.png"
run_exp "color_mnist" "1" "1" "3"

rm -f "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int2_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int2_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s1/test_acc_dim54_int2_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s1/test_acc_dim2048_int2_gen2_pseudo_prototypes_bic.png"
run_exp "color_mnist" "1" "2" "2"

rm -f "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int2_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int2_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s1/test_acc_dim54_int2_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s1/test_acc_dim2048_int2_gen3_pseudo_prototypes_bic.png"
run_exp "color_mnist" "1" "2" "3"

rm -f "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim54_int3_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s2/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s2/test_acc_dim54_int3_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s2/test_acc_dim2048_int3_gen2_pseudo_prototypes_bic.png"
run_exp "color_mnist" "2" "3" "2"

rm -f "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/color_mnist/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/color_mnist/s1/test_acc_dim54_int3_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/color_mnist/s1/test_acc_dim2048_int3_gen3_pseudo_prototypes_bic.png"
run_exp "color_mnist" "1" "3" "3"

rm -f "${LOG_ROOT}/covtype/s2/test_acc_dim54_int0_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim54_int0_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int0_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int0_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s2/test_acc_dim54_int0_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s2/test_acc_dim2048_int0_gen0_pseudo_prototypes_bic.png"
run_exp "covtype" "2" "0" "0"

rm -f "${LOG_ROOT}/covtype/s1/test_acc_dim54_int0_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim54_int0_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s1/test_acc_dim54_int0_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s1/test_acc_dim2048_int0_gen1_pseudo_prototypes_bic.png"
run_exp "covtype" "1" "0" "1"

rm -f "${LOG_ROOT}/covtype/s2/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim54_int0_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s2/test_acc_dim54_int0_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s2/test_acc_dim2048_int0_gen3_pseudo_prototypes_bic.png"
run_exp "covtype" "2" "0" "3"

rm -f "${LOG_ROOT}/covtype/s2/test_acc_dim54_int1_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim54_int1_gen0_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int1_gen0_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s2/test_acc_dim2048_int1_gen0_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s2/test_acc_dim54_int1_gen0_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s2/test_acc_dim2048_int1_gen0_pseudo_prototypes_bic.png"
run_exp "covtype" "2" "1" "0"

rm -f "${LOG_ROOT}/covtype/s0/test_acc_dim54_int1_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim54_int1_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s0/test_acc_dim54_int1_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s0/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic.png"
run_exp "covtype" "0" "1" "1"

rm -f "${LOG_ROOT}/covtype/s1/test_acc_dim54_int1_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim54_int1_gen1_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s1/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s1/test_acc_dim54_int1_gen1_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s1/test_acc_dim2048_int1_gen1_pseudo_prototypes_bic.png"
run_exp "covtype" "1" "1" "1"

rm -f "${LOG_ROOT}/covtype/s0/test_acc_dim54_int1_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim54_int1_gen2_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int1_gen2_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int1_gen2_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s0/test_acc_dim54_int1_gen2_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s0/test_acc_dim2048_int1_gen2_pseudo_prototypes_bic.png"
run_exp "covtype" "0" "1" "2"

rm -f "${LOG_ROOT}/covtype/s0/test_acc_dim54_int1_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim54_int1_gen3_pseudo_prototypes_bic_curves.jsonl" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic.txt" "${LOG_ROOT}/covtype/s0/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic_curves.jsonl"
rm -f "${PLOT_ROOT}/covtype/s0/test_acc_dim54_int1_gen3_pseudo_prototypes_bic.png" "${PLOT_ROOT}/covtype/s0/test_acc_dim2048_int1_gen3_pseudo_prototypes_bic.png"
run_exp "covtype" "0" "1" "3"

python collect_result.py --log-bases logs,logs_rerun
