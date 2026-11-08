"""Shared binary evaluation: class 1 is malignant."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def validate_test_data(frame, features):
    if frame.empty:
        raise ValueError('The CSV contains no records.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names are not supported.')
    missing = [name for name in features + ['target'] if name not in frame]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    X = frame[features].apply(pd.to_numeric, errors='raise')
    y = pd.to_numeric(frame['target'], errors='raise')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Features must be numeric, finite and non-missing.')
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Target must be 0 (benign) or 1 (malignant).')
    return X, y.astype(int)


def evaluate_model(model, X, y):
    prediction = model.predict(X)
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
    else:
        scores = model.decision_function(X)
    metrics = {
        'Accuracy': accuracy_score(y, prediction),
        'AUC': roc_auc_score(y, scores) if len(np.unique(y)) == 2 else float('nan'),
        'Precision': precision_score(y, prediction, zero_division=0),
        'Recall': recall_score(y, prediction, zero_division=0),
        'F1': f1_score(y, prediction, zero_division=0),
        'MCC': matthews_corrcoef(y, prediction)
    }
    return metrics, prediction
