#!/usr/bin/env bash
# Seeds 3-4 expansion for the headline JMLR tables (Path A bounded add-on, decided 2026-07-15).
# Brings headline configs to 5 seeds total. Mirrors run_portraits.sh / remaining_mnist.sh exactly,
# changing only the seed values. Rotation 90 deliberately excluded (appendix, near-random regime).
#
# RUN FROM THE GOAT REPO ROOT ON THE MACHINE THAT HOLDS ./data AND THE ENCODER CACHES
# (i.e., wherever seeds 0-2 were produced). Suggested launch:
#   nohup bash run_seeds_3_4.sh > seeds34_$(date +%m%d).log 2>&1 &
#
# Integration deadline (Path A): results integrate into the Friday draft ONLY if finished
# by Thursday 2026-07-16 night; otherwise they go to the revision cycle.

set -euo pipefail

LOG_ROOT="${LOG_ROOT:-logs_rerun}"
PLOT_ROOT="${PLOT_ROOT:-plots_rerun}"

seeds=(3 4)

# ---------------- Rotated MNIST (angles 30/45/60) ----------------
for s in "${seeds[@]}"; do
  for angle in 30 45 60; do
    # GOAT baseline (no generated domains)
    python experiment_refrac.py --rotation-angle "$angle" --seed "$s" --gt-domains 0 --generated-domains 0 --label-source pseudo --em-match prototypes --em-select bic --small-dim 2048
    # Generated-domain runs (both EM matching variants)
    python experiment_refrac.py --rotation-angle "$angle" --seed "$s" --gt-domains 0 --generated-domains 1 --label-source pseudo --em-match prototypes --em-select bic --small-dim 2048
    python experiment_refrac.py --rotation-angle "$angle" --seed "$s" --gt-domains 0 --generated-domains 1 --label-source pseudo --em-match pseudo --em-select bic --small-dim 2048
  done
done

# ---------------- Portraits ----------------
for s in "${seeds[@]}"; do
  # GOAT baseline canonical config (gen=0, em irrelevant -> prototypes)
  python experiment_refrac.py --plot-root "$PLOT_ROOT" --log-root "$LOG_ROOT" --dataset portraits --label-source pseudo --em-match prototypes --seed "$s" --gt-domains 0 --generated-domains 0
  # Generated-domain runs (both EM matching variants)
  python experiment_refrac.py --plot-root "$PLOT_ROOT" --log-root "$LOG_ROOT" --dataset portraits --label-source pseudo --em-match prototypes --seed "$s" --gt-domains 0 --generated-domains 1
  python experiment_refrac.py --plot-root "$PLOT_ROOT" --log-root "$LOG_ROOT" --dataset portraits --label-source pseudo --em-match pseudo --seed "$s" --gt-domains 0 --generated-domains 1
done

echo "All seed 3-4 headline runs complete."
