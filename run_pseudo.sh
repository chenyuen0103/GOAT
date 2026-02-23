# python experiment_new.py --rotation-angle 30 --label-source em
# python experiment_new.py --rotation-angle 45 --label-source em
# python experiment_new.py --rotation-angle 60 --label-source em
# python experiment_new.py --rotation-angle 75 --label-source em
# python experiment_new.py --rotation-angle 90 --label-source em

LOG_ROOT="${LOG_ROOT:-logs_rerun}"
PLOT_ROOT="${PLOT_ROOT:-plots_rerun}"

seeds=(1 2)
angles=(30 45 60 75 90)
gt_domains=(0 1 2 3)
generated_domains=(0 1 2 3)
small_dim=2048
em_select="bic"
em_match="pseudo"
label_source="pseudo"
em_ensemble_suffix=""

for s in "${seeds[@]}"; do
  for a in "${angles[@]}"; do
    for gt in "${gt_domains[@]}"; do
      for gd in "${generated_domains[@]}"; do
        log_base="test_acc_dim${small_dim}_int${gt}_gen${gd}_${label_source}_${em_match}_${em_select}${em_ensemble_suffix}.txt"
        log_dir="${LOG_ROOT}/mnist/s${s}/target${a}"
        log_path="${log_dir}/${log_base}"
        if [[ -f "$log_path" ]]; then
          echo "Skip (already ran): $log_path"
          continue
        fi
        python experiment_refrac.py --plot-root "$PLOT_ROOT" --log-root "$LOG_ROOT" --rotation-angle "$a" --label-source "$label_source" --seed "$s" --gt-domains "$gt" --generated-domains "$gd"
      done
    done
  done
done
