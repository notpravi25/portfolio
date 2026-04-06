"""
Step 2: Text Preprocessing
Stepwise Spam Detector - SMS Spam Collection
"""

import pandas as pd
import re

# ── Standard English Stopwords ─────────────────────────────────────────────────
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

# ── Rule-Based Lemmatizer ──────────────────────────────────────────────────────
def simple_lemmatize(word):
    rules = [
        ('ational','ate'), ('tional','tion'), ('enci','ence'), ('anci','ance'),
        ('izer','ize'), ('ising','ise'), ('izing','ize'), ('ised','ise'),
        ('ized','ize'), ('fulness','ful'), ('ousness','ous'), ('iveness','ive'),
        ('ings',''), ('ing',''), ('edly',''), ('ness',''), ('ment',''),
        ('tion',''), ('ions',''), ('ies','y'), ('ied','y'), ('eed','ee'),
        ('ed',''), ('ers','er'), ('es','e'), ('ly',''),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word

# ── Full Preprocessing Pipeline ────────────────────────────────────────────────
def preprocess(text):
    # Step 1: Lowercase
    text = text.lower()
    # Step 2: Remove punctuation, numbers, special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Step 3: Tokenize
    tokens = text.split()
    # Step 4: Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]
    # Step 5: Lemmatize
    tokens = [simple_lemmatize(t) for t in tokens]
    # Step 6: Remove very short tokens
    tokens = [t for t in tokens if len(t) >= 2]
    return ' '.join(tokens)

# ── Load Dataset ───────────────────────────────────────────────────────────────
df = pd.read_csv('../data/SMSSpamCollection.tsv', sep='\t',
                 header=None, names=['label', 'message'])

print(f"Dataset loaded: {df.shape[0]} messages")

# ── Apply Preprocessing ────────────────────────────────────────────────────────
df['cleaned'] = df['message'].apply(preprocess)
df['tokens']  = df['cleaned'].apply(lambda x: x.split())

# ── Preview Results ────────────────────────────────────────────────────────────
print("\n=== SAMPLE PREPROCESSING RESULTS ===")
for _, row in df.head(5).iterrows():
    print(f"[{row['label'].upper()}]")
    print(f"  BEFORE: {row['message'][:80]}")
    print(f"  AFTER : {row['cleaned'][:80]}")
    print()

# ── Save Preprocessed Dataset ─────────────────────────────────────────────────
df.to_csv('../data/dataset_preprocessed.csv', index=False)
print("Saved preprocessed dataset to data/dataset_preprocessed.csv")