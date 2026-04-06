"""
Step 4: Model Training - Naive Bayes
Stepwise Spam Detector - SMS Spam Collection
"""

import pandas as pd
import pickle
import scipy.sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ── Load Data ──────────────────────────────────────────────────────────────────
X_train = scipy.sparse.load_npz('../data/X_train.npz')
X_test  = scipy.sparse.load_npz('../data/X_test.npz')
y_train = pd.read_csv('../data/y_train.csv').squeeze()
y_test  = pd.read_csv('../data/y_test.csv').squeeze()

print(f"Training data : {X_train.shape}")
print(f"Test data     : {X_test.shape}")

# ── Train Naive Bayes ──────────────────────────────────────────────────────────
# MultinomialNB is ideal for text classification with TF-IDF features
model = MultinomialNB()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# ── Quick Accuracy Check ───────────────────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
print(f"\nTraining Accuracy : {train_acc * 100:.2f}%")
print(f"Test Accuracy     : {test_acc  * 100:.2f}%")

# ── What the model learned ─────────────────────────────────────────────────────
tfidf = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))
feature_names = tfidf.get_feature_names_out()

for i, cls in enumerate(model.classes_):
    top_indices = model.feature_log_prob_[i].argsort()[-10:][::-1]
    top_words   = [feature_names[j] for j in top_indices]
    print(f"\nTop 10 words model links to [{cls.upper()}]:")
    print(" ", top_words)

# ── Save Model ─────────────────────────────────────────────────────────────────
with open('../models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nSaved: models/spam_model.pkl")