"""
Step 6: Prediction on New Messages
Stepwise Spam Detector - SMS Spam Collection
"""

import pickle
import re

# ── Load Saved Model & Vectorizer ─────────────────────────────────────────────
model = pickle.load(open('../models/spam_model.pkl', 'rb'))
tfidf = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))

# ── Stopwords & Lemmatizer (same as Step 2) ───────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn'
}

def simple_lemmatize(word):
    rules = [
        ('ings',''),('ing',''),('edly',''),('ness',''),('ment',''),
        ('tion',''),('ions',''),('ies','y'),('ied','y'),('eed','ee'),
        ('ed',''),('ers','er'),('es','e'),('ly',''),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [simple_lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]
    return ' '.join(tokens)

def predict(message):
    """Predict whether a message is spam or ham."""
    cleaned    = preprocess(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    classes    = model.classes_
    spam_prob  = proba[list(classes).index('spam')] * 100
    ham_prob   = proba[list(classes).index('ham')]  * 100
    return prediction, spam_prob, ham_prob

# ── Predict on custom messages ────────────────────────────────────────────────
messages = [
    "Congratulations! You have won a FREE iPhone. Click now to claim your prize!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT: Your bank account has been suspended. Call 08001234567 immediately.",
    "Can you pick up some milk on your way home please?",
    "FREE entry! Text WIN to 87121 and receive £500 cash prize!",
    "I'll be late to the meeting, stuck in traffic. Start without me.",
]

print("=" * 65)
print("        SPAM DETECTOR — PREDICTION RESULTS")
print("=" * 65)

for msg in messages:
    pred, spam_prob, ham_prob = predict(msg)
    label = "SPAM" if pred == "spam" else "HAM"
    print(f"\nMessage : {msg[:70]}")
    print(f"Result  : [{label}]  |  Spam: {spam_prob:.1f}%  Ham: {ham_prob:.1f}%")
    print("-" * 65)

# ── Interactive Mode ───────────────────────────────────────────────────────────
print("\n\nTry your own message below (type 'quit' to exit):")
while True:
    user_input = input("\nEnter a message: ").strip()
    if user_input.lower() == 'quit':
        print("Exiting. Goodbye!")
        break
    if not user_input:
        print("Please enter a message.")
        continue
    pred, spam_prob, ham_prob = predict(user_input)
    label = "SPAM" if pred == "spam" else "HAM"
    print(f"Result  : [{label}]  |  Spam: {spam_prob:.1f}%  Ham: {ham_prob:.1f}%")