"""Run in BITS Virtual Lab. Generates models and measured results locally."""
from pathlib import Path
import json
import platform
from importlib.metadata import version, PackageNotFoundError
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from evaluation import evaluate_model

BASE = Path(__file__).resolve().parent
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'svm.joblib'
}


def prepare_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int).rename('target')
    return X, y, train_test_split(X, y, test_size=0.2,
                                 stratify=y, random_state=42)


def build_models():
    return {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))
    }


def train_and_save(X_train, X_test, y_train, y_test, output_dir=BASE):
    output_dir = Path(output_dir)
    (output_dir / 'model').mkdir(parents=True, exist_ok=True)
    models, rows, count_rows = build_models(), [], []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics, predicted = evaluate_model(model, X_test, y_test)
        rows.append({'ML Model Name': name, **metrics})
        tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
        count_rows.append({'ML Model Name': name, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})
        joblib.dump(model, output_dir / 'model' / MODEL_FILES[name])
        print('Trained and saved:', name)
    test = X_test.copy()
    test['target'] = y_test
    test.to_csv(output_dir / 'test_data.csv', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / 'model_comparison.csv', index=False)
    pd.DataFrame(count_rows).to_csv(output_dir / 'confusion_counts.csv', index=False)
    metadata = {'features': X_train.columns.tolist(), 'target': 'target',
                'labels': {'0': 'Benign', '1': 'Malignant'}, 'models': MODEL_FILES,
                'random_state': 42, 'training_records': len(X_train), 'test_records': len(X_test)}
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return models, results


def markdown_table(frame):
    rows = ['| ' + ' | '.join(frame.columns) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for _, row in frame.iterrows():
        rows.append('| ' + ' | '.join(str(v) if isinstance(v, str) else f'{v:.4f}' for v in row) + ' |')
    return '\n'.join(rows)


def export_environment(output_dir=BASE):
    output_dir = Path(output_dir)
    packages = ['streamlit', 'scikit-learn', 'numpy', 'pandas', 'scipy', 'matplotlib', 'joblib']
    pins = []
    for package in packages:
        try:
            pins.append(f'{package}=={version(package)}')
        except PackageNotFoundError:
            raise RuntimeError(f'Install {package} in the notebook kernel, then rerun this cell.')
    (output_dir / 'requirements.txt').write_text('\n'.join(pins) + '\n', encoding='utf-8')
    (output_dir / 'python_version.txt').write_text(platform.python_version() + '\n', encoding='utf-8')
    print('Pinned package versions. Training Python:', platform.python_version())


def main():
    X, y, split = prepare_data()
    print('Dataset:', X.shape, '| Missing values:', int(X.isna().sum().sum()))
    _, results = train_and_save(*split)
    (BASE / 'results_table.md').write_text(markdown_table(results), encoding='utf-8')
    export_environment()
    print(results.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
