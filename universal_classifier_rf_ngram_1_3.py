import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, average_precision_score, matthews_corrcoef
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from joblib import dump

def evaluate_model(y_true, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    return {
        "Acc": round(accuracy_score(y_true, y_pred), 3),
        "Prec": round(precision_score(y_true, y_pred), 3),
        "Rec": round(recall_score(y_true, y_pred), 3),
        "F1": round(f1_score(y_true, y_pred), 3),
        "Spec": round(specificity, 3),
        "Sens.": round(sensitivity, 3),
        "AUROC": round(roc_auc_score(y_true, y_pred_proba), 3),
        "AUPRC": round(average_precision_score(y_true, y_pred_proba), 3),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 3)
    }

# Load and split data
train_data = pd.read_csv('hemo_train.csv')
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=42)
test_data = pd.read_csv('hemo_test.csv')

# Create and train pipeline
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('ngram(1,3)', CountVectorizer(analyzer='char', ngram_range=(1, 3)), 'peptide_sequence')
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(train_set, train_set['label'])

# Evaluate
y_val_pred = pipeline.predict(val_set)
y_val_pred_proba = pipeline.predict_proba(val_set)[:, 1]
val_results = evaluate_model(val_set['label'], y_val_pred, y_val_pred_proba)

y_test_pred = pipeline.predict(test_data)
y_test_pred_proba = pipeline.predict_proba(test_data)[:, 1]
test_results = evaluate_model(test_data['label'], y_test_pred, y_test_pred_proba)

# Print results
print("\nValidation Set Results:", val_results)
print("\nTest Set Results:", test_results)

# Write results to file
with open('model_metrics.txt', 'w') as f:
    f.write("Validation Set Results:\n")
    for metric, value in val_results.items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nTest Set Results:\n")
    for metric, value in test_results.items():
        f.write(f"{metric}: {value}\n")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(test_data['label'], y_test_pred_proba)
roc_auc = auc(fpr, tpr)

fig = go.Figure()
hover_text = [f'Threshold: {threshold:.2f}<br>FPR: {fpr:.2f}<br>TPR: {tpr:.2f}' 
              for fpr, tpr, threshold in zip(fpr, tpr, thresholds)]

fig.add_trace(go.Scatter(
    x=fpr, 
    y=tpr, 
    mode='lines', 
    name=f'ROC curve (AUC = {roc_auc:.2f})',
    text=hover_text,
    hoverinfo='text'
))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
fig.update_layout(
    title='ROC Curve - RF with N-grams(1,3)',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(scaleanchor="x", scaleratio=1),
    xaxis=dict(constrain='domain'),
    width=700, height=500
)
fig.write_html("rf_ngram_auroc_plot.html")

# Save the trained model
dump(pipeline, 'imbalanced_mem_acp_rf_ngram_model.joblib')