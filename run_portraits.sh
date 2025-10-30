#!/usr/bin/env bash
set -euo pipefail



label_sources=(pseudo em)
em_matches=(prototypes pseudo)


for ls in "${label_sources[@]}"; do
  for m in "${em_matches[@]}"; do
    echo "Running: label_source=$ls, em_match=$m"
    python experiment_new.py --dataset portraits --label-source "$ls" --em-match "$m"
  done
done

