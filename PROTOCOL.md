# Prespecified analysis protocol (FedTGNN-SS revision)

Fixed on 26 September 2026, **before** any of the corrected experiments were run.
Any later deviation must be reported as such in the manuscript.

## Prediction task and time point
- **Primary task (GDM-early):** predict GDM at the first antenatal visit, before the 24–28-week OGTT, using only variables available at that visit. `OGTT` is excluded.
- **Secondary task (GDM-diagnostic):** the same cohort *with* OGTT. This is described as diagnostic classification, not early prediction.
- **Pima Indians** and **Early-Stage Diabetes Risk** are general-diabetes datasets. They are used only to show that the *method* generalises. They are **not** evidence of clinical validity for GDM.

## Data flow (per repeat r = 0..9, fold f = 0..4)
1. Stratified 5-fold outer split (seed 2026 + r). The test fold is never used for anything except the final evaluation.
2. 15% of the outer-training part is held out as a stratified **validation** set. It is used only to choose the decision threshold (maximum Youden's J).
3. The remaining training patients are split into S silos by a label-skew **Dirichlet(α·1)** partition (default α = 0.5). Every silo must hold at least 5 patients of each class (the draw is repeated otherwise).
4. Inside each silo, labels are removed **stratified by class, completely at random**. round((1−ρ)·n_c) labels are kept per class (at least 1). ρ ∈ {0.1, 0.3, 0.5, 0.7, 0.8} is the fraction of *training* labels removed *per silo*. Validation and test labels are never removed and never used for training.
5. Validation and test patients are assigned to a silo at random, with probability proportional to silo size. This is independent of their outcome.
6. Preprocessing (mean imputation and standardisation) is fitted on training patients only, from per-silo sufficient statistics.
7. Graphs are **inductive**. Each silo's k-NN graph contains only its training patients. An evaluation patient is attached by edges *from* its k nearest training patients in that silo. Training representations are provably unchanged (unit test).
8. Splits and silos depend on (r, f) only, so all methods and all ρ are paired.

## Hyperparameters
- Fixed **a priori** from the original implementation (single global configuration, `fedtgnn/fedtgnn.py::FedTGNNConfig`).
- **No hyperparameter was tuned on any data in this study**, for any method. All baselines use library or published defaults (zero tuning budget for every method).
- Sensitivity to the main hyperparameters is reported separately (`run.py sweep`). These results are descriptive and are not used to select a configuration.

## Endpoints
- **Primary endpoint:** test AUROC on GDM-early at ρ = 0.8 (α = 0.5, S = 3).
- **Primary comparison:** FedTGNN-SS vs FedAvg-LR.
- **Secondary endpoints:** AUPRC, macro-F1, sensitivity, specificity, PPV, NPV (at the validation-selected threshold), Brier score, calibration intercept and slope, ECE, confusion counts, and net benefit (decision curves), for all datasets and ρ.

## Statistics
- Unit of analysis: paired (repeat, fold) results, n = 50 per comparison in the main experiment.
- Mean paired difference with a 95% CI and p-value from the **Nadeau–Bengio corrected resampled t-test** (primary inferential method). The two-sided Wilcoxon signed-rank p-value is reported as a sensitivity analysis.
- **Holm** adjustment within each family. A family is one dataset × one metric, covering all baselines × all ρ.
- Effect size: matched-pairs rank-biserial correlation.
- "Significant" means Holm-adjusted p < 0.05. Clinical relevance is judged separately, using the size of the difference and its CI.

## Additional experiments (secondary, descriptive)
- Heterogeneity: α ∈ {0.1, 0.5, 1.0, IID} at ρ = 0.8 (5 repeats × 5 folds).
- Number of clients: S ∈ {2, 3, 4, 5} on GDM-early at ρ = 0.8.
- Ablation of every component at ρ ∈ {0.5, 0.8} on all datasets.
- Hyperparameter sensitivity at ρ = 0.8 on GDM-early and Pima.
- Pseudo-label precision and coverage per round (diagnostic use of hidden labels only).
- Privacy audit: loss-threshold membership-inference AUROC for FedTGNN-SS and FedAvg-LR. No formal differential-privacy guarantee is claimed.
- Cost: communicated bytes and training time per method.

## Amendment 1 (26 September 2026, before any primary-endpoint result was analysed)
Measured run time was about 110 s of single-core CPU per GDM fold for all 18 methods. The full grid was therefore cut down for feasibility. The primary analysis is **unchanged**: GDM-early, Pima and Early-Stage keep 10 repeats × 5 folds. The secondary analyses were reduced:
- GDM-diagnostic (with OGTT): 3 × 5.
- Ablation: 5 × 5 (Pima, Early) and 3 × 5 (GDM-early).
- Heterogeneity: 5 × 5 and 3 × 5.
- Number of clients: 3 × 5.
- Sweeps: 3 × 5 (Pima) and 2 × 5 (GDM-early).

Only single-fold smoke tests had been inspected at the time of this amendment.

## Amendment 2 (27 September 2026, after the primary results were known): exploratory analyses
These two analyses were added **after** the main results had been examined. They are reported as exploratory and are not used for any confirmatory claim.
1. **GDM cohort characteristics** (`gdm_ceiling.py`):
   - baseline table by outcome;
   - single-predictor AUROC (logistic regression, same 10 × 5 folds);
   - leave-one-predictor-out AUROC.
2. **Leakage experiment** (`run.py leakage`): FedTGNN-SS under four evaluation shortcuts, each alone and all combined, on the same partitions:
   - L1: test patients in the training graph;
   - L2: a pooled final classifier;
   - L3: preprocessing fitted on all data;
   - L4: one cross-silo graph at inference.

   FedAvg-LR under L3 is included as a reference. Run with 5 repeats × 5 folds at ρ ∈ {0.1, 0.8} on GDM-early, Pima and Early-Stage. Paired differences from the correct protocol are summarised with 95% CIs (corrected resampled t-test).
