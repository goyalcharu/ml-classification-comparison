"""Inference only: uploaded CSVs are never used for fitting models."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from evaluation import evaluate_model, validate_test_data

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Diagnostic Model Explorer', page_icon='📊', layout='wide')
st.title('Diagnostic Model Explorer')
st.caption('Machine Learning Assignment 2 | Charu Goyal | 2026AJ05682')
st.write('Compare five trained classifiers on labelled breast cancer test records.')

@st.cache_resource
def load_artifacts():
    metadata = json.loads((BASE / 'metadata.json').read_text(encoding='utf-8'))
    models = {name: joblib.load(BASE / 'model' / file)
              for name, file in metadata['models'].items()}
    return metadata, models

try:
    metadata, models = load_artifacts()
except Exception as error:
    st.error('Model loading failed. Run the notebook, upload its generated model files '
             'and metadata.json, and match the training package versions.')
    st.code(str(error))
    st.stop()

selected = st.sidebar.selectbox('Model to inspect', list(models))
st.sidebar.info('0 = Benign; 1 = Malignant. Models are loaded, never retrained here.')
with st.expander('CSV requirements'):
    st.write('Upload held-out test data with all listed features and a target column. '
             'Values must be numeric and non-missing; target must be 0 or 1.')
    st.write(metadata['features'] + ['target'])
if (BASE / 'test_data.csv').exists():
    st.download_button('Download example test CSV', (BASE / 'test_data.csv').read_bytes(),
                       file_name='test_data.csv', mime='text/csv')
upload = st.file_uploader('Upload labelled test CSV', type=['csv'])
# The same input renderer is used for the bundled sample and for uploads.
use_sample = st.checkbox('Evaluate bundled test data', value=False)
if upload is None and not use_sample:
    st.info('Upload your test CSV, or select the bundled test data to explore the results.')
    st.stop()
try:
    frame = pd.read_csv(upload if upload is not None else BASE / 'test_data.csv')
    X, y = validate_test_data(frame, metadata['features'])
except Exception as error:
    st.error(f'Invalid test data: {error}')
    st.stop()
st.success(f'{len(frame)} test records validated | 30 input features')
if y.nunique() < 2:
    st.warning('Only one target class is present: AUC is undefined (N/A). '
               'Precision, recall and F1 use zero when their denominator is zero.')
rows, predictions = [], {}
for name, model in models.items():
    metrics, prediction = evaluate_model(model, X, y)
    rows.append({'ML Model Name': name, **metrics})
    predictions[name] = prediction
comparison = pd.DataFrame(rows).set_index('ML Model Name')
st.subheader('All models on the same test records')
st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'))
st.download_button('Download metrics', comparison.to_csv().encode(),
                   file_name='evaluation_results.csv', mime='text/csv')
st.subheader(selected)
columns = st.columns(3)
for i, (metric, value) in enumerate(comparison.loc[selected].items()):
    columns[i % 3].metric(metric, 'N/A' if pd.isna(value) else f'{value:.4f}')
left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, predictions[selected], labels=[0, 1],
        display_labels=['Benign', 'Malignant'], cmap='Blues', colorbar=False, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    report = classification_report(y, predictions[selected], labels=[0, 1],
        target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))
with st.expander('Preview evaluated data'):
    st.dataframe(frame.head(10))
st.caption('Academic evaluation only. Recall refers to malignant samples. '
           'Performance on this small test set does not establish clinical suitability.')
