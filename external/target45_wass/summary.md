# Rotated MNIST Label-Prior Shift Summary

Run configuration:
- Output directory: `analysis_outputs/rmnist_label_shift_wass_spectrum`
- Seeds: 10 (`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`)
- Source/target samples per setting: 1000 / 1000
- Encoder setup: `small_dim=20`, `source_epochs=5`
- Adaptation setup: `generated_domains=2`, `adapt_epochs=3`
- Methods: `cgda_fr`, `cgda_fr_oracle`, `cgda_wass`, `goat`

Completed outputs:
- `results.csv`: 840 method rows across 210 condition/skew/seed settings.
- `diagnostics.csv`: 210 setting-level diagnostic rows.
- `per_class_recall.csv`: 8400 method/class rows.
- `figures/`: balanced risk, GOAT CrossOT, CGDA class-structure error, rarest-class recall, CGDA-Wass mechanism plots, oracle gap, and oracle-vs-estimated CGDA diagnostic plots.
- `validation.json`: passed; missing result rows = 0, Delta p max error = 0.00.

## 1. Does GOAT cross-class mixing grow with Delta p?
Yes. In label-only shift, mean GOAT CrossOT increased from 0.188 at Delta p = 0.00 to 0.850 at Delta p = 1.62. The per-run Pearson correlation between Delta p and GOAT CrossOT was 0.996.
In combined shift, CrossOT was already high because of feature shift, but it still increased from 0.698 at Delta p = 0.00 to 0.880 at Delta p = 1.62. The per-run Pearson correlation was 0.972.
Feature-only shift had Delta p = 0 by construction and mean CrossOT = 0.701, showing that feature rotation alone can create substantial pooled OT cross-class matches even without label-prior drift.

## 2. Does balanced risk degrade accordingly?
Mostly, but less cleanly than CrossOT. For GOAT, label-only balanced risk increased from 0.378 at Delta p = 0.00 to 0.507 at Delta p = 1.62. Combined-shift balanced risk increased from 0.749 to 0.780 across the same Delta p range.
The risk trend is weaker than the CrossOT trend because the RMNIST classifier and adaptation dynamics add variance, and because combined shift already starts from high risk at Delta p = 0.

## 3. Does CGDA reduce this class-mixing channel?
Fair comparison is GOAT vs estimated CGDA-FR. Positive GOAT-minus-CGDA gap means estimated CGDA-FR has lower balanced risk. label: mean gap 0.032, CGDA-FR lower risk on 68% of paired settings; combined: mean gap 0.007, CGDA-FR lower risk on 58% of paired settings; feature: mean gap 0.018, CGDA-FR lower risk on 80% of paired settings.
For label-only shift, the fair risk gap moved from 0.117 at skew 0.0 to -0.057 at skew 0.9, so estimated CGDA-FR loses its advantage at the highest skew levels.
For combined shift, the fair risk gap moved from 0.027 at skew 0.0 to 0.005 at skew 0.9, so estimated CGDA-FR loses its advantage at the highest skew levels.

## CGDA-Wass Mechanism Comparison
For the combined 0->45 condition, CGDA-Wass minus GOAT balanced-accuracy gap averaged 0.006 overall, 0.017 over moderate skew 0.2-0.5, and -0.014 over high skew >= 0.7. CGDA-Wass beat GOAT on 56% of combined paired settings.
At combined skew 0.0, mean balanced accuracy was GOAT 0.251, CGDA-Wass 0.276, and CGDA-FR 0.278.
At combined skew 0.5, mean balanced accuracy was GOAT 0.241, CGDA-Wass 0.254, and CGDA-FR 0.258.
At combined skew 0.9, mean balanced accuracy was GOAT 0.220, CGDA-Wass 0.213, and CGDA-FR 0.225.
CGDA-Wass and CGDA-FR share the same target class recovery. In combined high skew >= 0.7, CGDA-Wass minus CGDA-FR balanced-accuracy gap averaged -0.004.
For CGDA-Wass in combined shift, balanced risk correlated with shared class-structure error at r=0.365, so failures should be read through target class-recovery quality rather than a pooled-OT channel.
The diagnostic supports the mechanism: CGDA-FR does not use pooled source-target OT matching as its generation channel, and the oracle CGDA-FR rows show that class-conditional generation can be substantially better when target class structure is known. The mean estimated-minus-oracle CGDA-FR balanced-risk gap was 0.093.
In label-only shift, oracle CGDA-FR balanced risk increased from 0.231 to 0.403, while estimated CGDA-FR increased from 0.261 to 0.564. This separates the class-conditional generation benefit from the target class-estimation bottleneck.

## 4. When does CGDA fail?
CGDA-FR fails when the unsupervised target class estimate degrades. Class-structure error increased with skew: in label-only shift it rose from 0.325 at Delta p = 0.00 to 0.757 at Delta p = 1.62; in combined shift it rose from 0.598 to 0.782.
The lowest observed target-cluster minority recovery was 0.000 at condition `label`, target angle 0, skew 0.9, seed 7. That run had cluster balanced accuracy 0.154 and class-structure error 0.846.

Overall interpretation: RMNIST validates the qualitative mechanism. Pooled GOAT transport mixes true classes more as label-prior distance grows, especially in label-only and combined settings. CGDA-FR removes the pooled OT class-mixing channel, but its practical bottleneck is reliable unsupervised recovery of target class structure, especially under extreme skew and rotation.
