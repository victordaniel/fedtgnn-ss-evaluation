"""Dataset loaders.

Every loader returns *raw* (unimputed, unscaled) features so that all
preprocessing can be fitted on training data only (see preprocess.py).
"""

import io
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, '..', 'data_cache')

# Download GDM.xlsx from Kaggle (sumathisanthosh/gestational-diabetes-mellitus-gdm-data-set)
# into ../data/, or point the GDM_XLSX environment variable at it.
DEFAULT_GDM_PATH = os.environ.get('GDM_XLSX', os.path.join(HERE, '..', 'data', 'GDM.xlsx'))

GDM_TARGET = 'Class Label(GDM /Non GDM)'

# Variables that are the diagnostic test itself. They are excluded from the
# early-prediction feature set (prediction time point: first antenatal visit,
# before the 24-28 week OGTT).
GDM_DIAGNOSTIC_VARS = ['OGTT']

PIMA_URL = ('https://raw.githubusercontent.com/jbrownlee/Datasets/master/'
            'pima-indians-diabetes.data.csv')
PIMA_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
# Physiologically impossible zeros are coded missing (standard practice).
PIMA_ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness',
                        'Insulin', 'BMI']

EARLY_URL = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
             '00529/diabetes_data_upload.csv')


def _cached_csv(name, url, **read_kw):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        with open(path, 'wb') as f:
            f.write(raw)
    return pd.read_csv(path, **read_kw)


def load_gdm(feature_set='early', path=DEFAULT_GDM_PATH):
    """GDM dataset.

    feature_set='early'      : excludes OGTT (primary analysis)
    feature_set='diagnostic' : includes OGTT (secondary, diagnostic setting)
    """
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in ['Case Number'] if c in df.columns])
    y = df[GDM_TARGET].astype(int).values
    X = df.drop(columns=[GDM_TARGET])
    if feature_set == 'early':
        X = X.drop(columns=GDM_DIAGNOSTIC_VARS)
    elif feature_set != 'diagnostic':
        raise ValueError(feature_set)
    return X.values.astype(float), y, list(X.columns)


def load_pima():
    df = _cached_csv('pima.csv', PIMA_URL, header=None, names=PIMA_COLS)
    y = df['Outcome'].astype(int).values
    X = df.drop(columns=['Outcome']).astype(float)
    for c in PIMA_ZERO_AS_MISSING:
        X.loc[X[c] == 0, c] = np.nan
    return X.values, y, list(X.columns)


def load_early():
    df = _cached_csv('early_stage.csv', EARLY_URL)
    y = (df['class'].astype(str).str.strip() == 'Positive').astype(int).values
    X = df.drop(columns=['class']).copy()
    X['Gender'] = (X['Gender'].astype(str).str.strip() == 'Male').astype(float)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = (X[c].astype(str).str.strip() == 'Yes').astype(float)
    return X.values.astype(float), y, list(X.columns)


def load(name):
    if name == 'gdm_early':
        return load_gdm('early')
    if name == 'gdm_diag':
        return load_gdm('diagnostic')
    if name == 'pima':
        return load_pima()
    if name == 'early':
        return load_early()
    raise ValueError(name)
