"""
Step 3: Feature Extraction using TF-IDF
Stepwise Spam Detector - SMS Spam Collection
"""

import pandas as pd
import pickle
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── Load Preprocessed Data ─────────────────────────────────────────────────────
df = pd.read_csv('../data/dataset_preprocessed.csv')
df['cleaned'] = df['cleaned'].fillna('')

print(f"Loaded {len(df)} messages")

# ── TF-IDF Vectorization ───────────────────────────────────────────────────────
# max_features=3000 → keep only the top 3000 most meaningful words
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])   # shape: (5572, 3000)
y = df['label']

print(f"\n=== FEATURE MATRIX ===")
print(f"Shape            : {X.shape}  (messages x features)")
print(f"Non-zero entries : {X.nnz}  (most values are 0 - sparse matrix)")

# ── Train / Test Split ─────────────────────────────────────────────────────────
# 80% training, 20% testing
# stratify=y → keeps spam/ham ratio same in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n=== TRAIN / TEST SPLIT ===")
print(f"Training : {X_train.shape[0]} messages")
print(f"Testing  : {X_test.shape[0]} messages")
print(f"\nTraining distribution:\n{y_train.value_counts()}")
print(f"\nTesting distribution:\n{y_test.value_counts()}")

# ── Save Everything for Step 4 ────────────────────────────────────────────────
scipy.sparse.save_npz('../data/X_train.npz', X_train)
scipy.sparse.save_npz('../data/X_test.npz',  X_test)
pd.Series(y_train.values).to_csv('../data/y_train.csv', index=False)
pd.Series(y_test.values).to_csv('../data/y_test.csv',   index=False)

with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\n✅ Saved: data/X_train.npz")
print("✅ Saved: data/X_test.npz")
print("✅ Saved: data/y_train.csv")
print("✅ Saved: data/y_test.csv")
print("✅ Saved: models/tfidf_vectorizer.pkl")