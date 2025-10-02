#!/usr/bin/env bash
set -euo pipefail

angles=(45 60 90)
# angles=(15)
label_sources=(em pseudo)
em_matches=(pseudo prototypes)

for a in "${angles[@]}"; do
  for ls in "${label_sources[@]}"; do
    for m in "${em_matches[@]}"; do
      echo "Running: angle=$a, label_source=$ls, em_match=$m"
      python experiment_new.py --rotation-angle "$a" --label-source "$ls" --em-match "$m"
    done
  done
done
