#!/usr/bin/env bash
# Runs every prespecified experiment (see PROTOCOL.md, incl. Amendment 1),
# then the statistics and tables. Resumable: finished rows are skipped.
set -e
cd "$(dirname "$0")"
W=${WORKERS:-3}
# --- main comparison (primary + secondary endpoints)
python run.py main --datasets gdm_early pima early --repeats 10 --workers "$W"
python stats.py results/main.csv --out results/stats_main.csv
python run.py main --datasets gdm_diag --repeats 3 --workers "$W"
# --- ablation
python run.py ablation --datasets pima early --repeats 5 --workers "$W"
python run.py ablation --datasets gdm_early --repeats 3 --workers "$W"
# --- heterogeneity and number of clients (rho = 0.8)
python run.py hetero --datasets pima early --repeats 5 --workers "$W"
python run.py hetero --datasets gdm_early --repeats 3 --workers "$W"
python run.py clients --datasets gdm_early --repeats 3 --workers "$W"
# --- hyperparameter sensitivity (descriptive)
python run.py sweep --datasets pima --repeats 3 --workers "$W"
python run.py sweep --datasets gdm_early --repeats 2 --workers "$W"
python stats.py results/main.csv --out results/stats_main.csv
python make_tables.py
echo ALL DONE
