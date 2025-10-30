#!/usr/bin/env bash
set -euo pipefail

angles=(60)
angles=(45 90 30 60)
label_sources=(pseudo em)
em_matches=(prototypes pseudo)

for a in "${angles[@]}"; do
  for ls in "${label_sources[@]}"; do
    for m in "${em_matches[@]}"; do
      echo "Running: angle=$a, label_source=$ls, em_match=$m"
      python experiment_new.py --rotation-angle "$a" --label-source "$ls" --em-match "$m"
    done
  done
done
