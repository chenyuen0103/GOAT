#!/usr/bin/env bash
set -euo pipefail

label_sources=(pseudo)
em_matches=(prototypes)
seeds=(0 1 2)
gt_domains=(0 1 2 3)
generated_domains=(0)
dataset="covtype"
# CovType's MLP encoder outputs 54-d features, and experiment_refrac.py will clamp
# --small-dim down to that. Set it explicitly so log filenames match and skipping works.
small_dim=54
em_select="bic"
em_ensemble_suffix=""

# If 1, rerun gen=0 experiments even if logs already exist (old results were saved differently).
RERUN_GEN0="${RERUN_GEN0:-1}"

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
          log_dir="logs/${dataset}/s${s}"
          log_path="${log_dir}/${log_base}"
          if [[ -f "$log_path" ]]; then
            if [[ "$gd" == "0" && "$RERUN_GEN0" == "1" ]]; then
              # Only force a gen=0 rerun once (when no prior backups exist).
              shopt -s nullglob
              backups=("${log_path}.bak."*)
              shopt -u nullglob
              if (( ${#backups[@]} > 0 )); then
                echo "Skip (gen=0 already backed up): $log_path"
                continue
              fi

              ts="$(date +%Y%m%d_%H%M%S)"
              echo "Rerun gen=0: backing up existing logs -> ${log_path}.bak.${ts}"
              mv -f "$log_path" "${log_path}.bak.${ts}"
              curves_path="${log_path%.txt}_curves.jsonl"
              if [[ -f "$curves_path" ]]; then
                mv -f "$curves_path" "${curves_path}.bak.${ts}"
              fi
            else
              echo "Skip (already ran): $log_path"
              continue
            fi
          fi
          echo "Running: dataset=covtype, label_source=$ls, em_match=$m, seed=$s, gt_domains=$gt, generated_domains=$gd"
          python experiment_refrac.py --dataset "$dataset" --label-source "$ls" --em-match "$m" --seed "$s" --gt-domains "$gt" --generated-domains "$gd" --small-dim "$small_dim"
        done
      done
    done
  done
done
