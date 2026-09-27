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

## Amendment 3 (27 September 2026, after the primary results were known): exploratory recalibration
FedTGNN-SS and FedAvg-LR were re-run at ρ = 0.8 (10 × 5 folds; same partitions and seeds), with validation predictions stored (`run.py recal`). Platt scaling and temperature scaling are fitted on the validation predictions only and applied to the test fold (`recalibrate.py`). This analysis is exploratory and does not alter any primary result. Recalibration by a monotone map leaves AUROC unchanged.

## Amendment 4 (27 September 2026, after the primary results were known): exploratory graph diagnostics
`graph_analysis.py` describes the quality of the initial feature-space graph and of the refined (AGR) embedding graph:
- edge homophily against its random baseline;
- node purity, overall and by class;
- the share of unlabelled patients with a labelled neighbour, and agreement with the labelled neighbours.

It also evaluates a federated distance-weighted k-NN ensemble. The withheld outcomes are used only to describe the graphs, never for training. Runs: 3 repeats × 5 folds at ρ ∈ {0.1, 0.8}. Exploratory.

## Amendment 5: prospectively specified sensitivity analyses, written after the primary results were known
Written and committed on 27 September 2026, **before** either analysis below was run. These analyses were **not** part of the original prespecified protocol. They were specified in advance of their own results, after the primary results were known, in response to anticipated reviewer objections. Neither alters the primary analysis or its conclusion rule.

### 5a. Informative (risk-dependent) label missingness (`run.py mar`)
- **Mechanism.** Within each silo, patient *i* keeps its training label with probability sigmoid(a + β·z_i).
  - z_i is the silo-standardised risk score: the mean of standardised established risk factors, each positively associated with the outcome a priori.
  - a is solved numerically so that the expected labelled fraction equals 1 − ρ.
  - Labels are Bernoulli draws, with at least one labelled patient per class guaranteed.
  - The outcome is never used to define z.
- **Risk factors.**
  - GDM-early: Age, BMI, Family History, PCOS, Prediabetes, Large Child or Birth Default. **OGTT is not used.**
  - Pima: Glucose, BMI, Age.
  - Early-Stage: Polyuria, Polydipsia, Age.
- **Strengths:** β = 1.0 (moderate) and β = 2.5 (strong). Everything else is identical to the main study: Dirichlet α = 0.5, S as in the main study, validation hold-out, inductive graphs, and the threshold rule.
- **Scarcity:** ρ = 0.8.
- **Runs:** 5 repeats × 5 folds, with the main-study partition seeds (`2026 + 1000 r + f`).
- **Methods:** FedTGNN-SS, FedAvg-LR, FedAvg-MLP, FedEns-RF, FedAvg-GCN, FedMatch-tab, Local-TGNN.
- **Endpoints:** test AUROC, Brier score and calibration slope. The comparison is FedTGNN-SS minus each comparator, with the paired difference and a 95% CI from the corrected resampled t-test. Results are reported descriptively, without significance claims.

### 5b. Baseline tuning sensitivity (`run.py tuned`)
- **Methods tuned:** FedAvg-LR, FedAvg-MLP, FedEns-RF, FedEns-XGB, FedEns-SVM.
- **Grids** (fixed; `fedtgnn/tuned.py`):
  - LR: C ∈ {0.01, 0.1, 1, 10}.
  - MLP: hidden ∈ {32, 64, 128} × learning rate ∈ {0.001, 0.005, 0.01}.
  - RF: max_depth ∈ {None, 5, 10} × min_samples_leaf ∈ {1, 5}.
  - XGB: max_depth ∈ {3, 6} × n_estimators ∈ {100, 300} × learning_rate ∈ {0.05, 0.3}.
  - SVM: C ∈ {0.1, 1, 10} × gamma ∈ {scale, 0.01, 0.1}.
- **Selection rule:** the configuration with the highest AUROC on the validation set of the same repeat and fold. Ties go to the first configuration in grid order. The test fold is never consulted. The decision threshold is then chosen on the same validation set, as in the main study.
- **Scarcity:** ρ ∈ {0.1, 0.8}.
- **Runs:** 5 repeats × 5 folds, with the main-study partitions.
- **Comparison:** FedTGNN-SS from the main study, on the identical repeat/fold partitions (repeats 0–4), minus each tuned baseline. Test AUROC, with the paired difference and a 95% CI from the corrected resampled t-test. Descriptive.
- **Note on fairness:** the FedTGNN-SS configuration was inherited from earlier work on these datasets, which may have influenced it. This analysis gives the main baselines an equal, limited, validation-only tuning budget.

### Stopping rule
After Amendments 3–5 (recalibration, graph diagnostics, 5a, 5b), no further experiments will be added, unless one of them reveals a substantive methodological problem.
