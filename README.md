# FedTGNN-SS: corrected, leakage-controlled experiment code

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22987367.svg)](https://doi.org/10.5281/zenodo.22987367)

Code for *"Federated Semi-Supervised Graph Neural Networks for the Prediction of Gestational and Other Diabetes from Tabular Records: A Leakage-Controlled Evaluation"*.

## Layout
| Path | Content |
|---|---|
| `fedtgnn/data.py` | loaders (GDM early/diagnostic, Pima, Early-Stage); raw features only |
| `fedtgnn/splits.py` | repeated CV → validation hold-out → Dirichlet silos → per-silo stratified label masking |
| `fedtgnn/preprocess.py` | federated mean-imputation and standardisation from per-silo sufficient statistics |
| `fedtgnn/graph.py` | local k-NN graphs (training patients only), inductive insertion of evaluation patients |
| `fedtgnn/context.py` | per-silo tensors; unlabeled outcomes are masked to −1 |
| `fedtgnn/fedtgnn.py` | FedTGNN-SS with switchable components (`FedTGNNConfig`) |
| `fedtgnn/heads.py` | logistic regression by FedAvg / FedProx / local / central |
| `fedtgnn/baselines.py` | the 17 comparators |
| `fedtgnn/metrics.py` | threshold selection, discrimination, calibration, net benefit, membership inference |
| `fedtgnn/variants.py` | ablation variants and sensitivity sweeps |
| `run.py` | resumable experiment runner (`main`, `hetero`, `clients`, `ablation`, `sweep`) |
| `stats.py` | prespecified statistics (Nadeau–Bengio corrected t-test, Wilcoxon, Holm) |
| `make_tables.py` | writes every manuscript table, figure and number macro to `../mdpi/generated/` |
| `tests/` | leakage, count and metric unit tests |
| `PROTOCOL.md` | prespecified analysis plan (+ Amendment 1) |

## Data
No data are redistributed here.
- **GDM:** download `GDM.xlsx` from Kaggle, [*Gestational Diabetes Mellitus (GDM Data Set)*](https://www.kaggle.com/datasets/sumathisanthosh/gestational-diabetes-mellitus-gdm-data-set) (S. Santhosh; licence CC BY-NC-SA 4.0). Place it in `data/GDM.xlsx`, or set `GDM_XLSX`.
- **Pima Indians Diabetes** and **Early-Stage Diabetes Risk** (UCI) are downloaded automatically into `data_cache/`.

## Reproduce
```bash
pip install torch torch_geometric scikit-learn xgboost scipy pandas openpyxl matplotlib pytest
python -m pytest tests -q     # leakage / protocol checks
bash run_all.sh               # ~10 h on a 4-core CPU; resumable
```
`results/` contains the run-level metrics used in the paper. The per-patient test predictions (`main_preds.csv.gz`, used by `make_tables.py` for the calibration figure and by `repair_main.py`) are not included; `run.py main` regenerates them.

## Citation
Daniel, G.V.; M, V. *Federated Semi-Supervised Graph Neural Networks for the Prediction of Gestational and Other Diabetes from Tabular Records: A Leakage-Controlled Evaluation.* Manuscript submitted to *Diagnostics* (MDPI), 2026.

Software: Daniel, G.V.; M, V. *FedTGNN-SS: leakage-controlled evaluation* (v1.0.0). Zenodo, 2026. https://doi.org/10.5281/zenodo.22987367

Seeds: the split for repeat *r*, fold *f* uses seed `2026 + 1000 r + f`. Every row of `results/*.csv` stores its seed and per-silo counts.
