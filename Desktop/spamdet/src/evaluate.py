"""
Step 5: Model Evaluation
Stepwise Spam Detector - SMS Spam Collection
"""

import pandas as pd
import pickle
import scipy.sparse
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ── Load Data & Model ──────────────────────────────────────────────────────────
X_test  = scipy.sparse.load_npz('../data/X_test.npz')
y_test  = pd.read_csv('../data/y_test.csv').squeeze()
model   = pickle.load(open('../models/spam_model.pkl', 'rb'))

# ── Predictions ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Metrics ────────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='spam')
rec  = recall_score(y_test, y_pred, pos_label='spam')
f1   = f1_score(y_test, y_pred, pos_label='spam')
cm   = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])

print("=" * 45)
print("       MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"  Accuracy  : {acc  * 100:.2f}%")
print(f"  Precision : {prec * 100:.2f}%")
print(f"  Recall    : {rec  * 100:.2f}%")
print(f"  F1-Score  : {f1   * 100:.2f}%")
print("=" * 45)

print("\n=== CONFUSION MATRIX ===")
print(f"                  Predicted HAM   Predicted SPAM")
print(f"  Actual HAM  :       {cm[0][0]}              {cm[0][1]}")
print(f"  Actual SPAM :        {cm[1][0]}             {cm[1][1]}")

print("\n=== FULL CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# ── Error Summary ──────────────────────────────────────────────────────────────
tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
print("=== ERROR BREAKDOWN ===")
print(f"  True Positives  (spam caught correctly) : {tp}")
print(f"  True Negatives  (ham correct)           : {tn}")
print(f"  False Positives (ham flagged as spam)   : {fp}")
print(f"  False Negatives (spam that slipped)     : {fn}")