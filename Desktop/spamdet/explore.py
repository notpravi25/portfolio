"""
Step 1: Dataset Exploration
Stepwise Spam Detector - SMS Spam Collection
"""

import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
df = pd.read_csv('../data/SMSSpamCollection.tsv', sep='\t', header=None, names=['label', 'message'])

# ── 2. Basic Structure ─────────────────────────────────────────────────────────
print("=" * 50)
print("DATASET SHAPE:", df.shape)
print("=" * 50)
print(df.head())
print("\nData Types:\n", df.dtypes)
print("\nNull Values:\n", df.isnull().sum())

# ── 3. Label Distribution ──────────────────────────────────────────────────────
print("\n=== LABEL DISTRIBUTION ===")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True).mul(100).round(2))

# ── 4. Message Length & Word Count ────────────────────────────────────────────
df['msg_length'] = df['message'].apply(len)
df['word_count']  = df['message'].apply(lambda x: len(x.split()))

print("\n=== MESSAGE LENGTH STATS (by label) ===")
print(df.groupby('label')['msg_length'].describe().round(2))

print("\n=== WORD COUNT STATS (by label) ===")
print(df.groupby('label')['word_count'].describe().round(2))

# ── 5. Sample Messages ────────────────────────────────────────────────────────
print("\n=== SAMPLE SPAM MESSAGES ===")
for _, row in df[df['label'] == 'spam'].head(3).iterrows():
    print(f"  » {row['message'][:120]}")

print("\n=== SAMPLE HAM MESSAGES ===")
for _, row in df[df['label'] == 'ham'].head(3).iterrows():
    print(f"  » {row['message'][:120]}")

# ── 6. Visualizations ─────────────────────────────────────────────────────────
colors = {'ham': '#4CAF50', 'spam': '#F44336'}
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Step 1 – Dataset Exploration: SMS Spam Collection', fontsize=14, fontweight='bold')

# Pie chart
counts = df['label'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=[colors['ham'], colors['spam']], startangle=90)
axes[0].set_title('Spam vs Ham Distribution', fontweight='bold')

# Message length histogram
for label, grp in df.groupby('label'):
    axes[1].hist(grp['msg_length'], bins=40, alpha=0.6, label=label, color=colors[label])
axes[1].set_title('Message Length Distribution', fontweight='bold')
axes[1].set_xlabel('Characters')
axes[1].set_ylabel('Count')
axes[1].legend()

# Word count box plot
bp = axes[2].boxplot([df[df['label']=='ham']['word_count'],
                      df[df['label']=='spam']['word_count']],
                     tick_labels=['Ham', 'Spam'], patch_artist=True)
bp['boxes'][0].set_facecolor('#4CAF50')
bp['boxes'][1].set_facecolor('#F44336')
axes[2].set_title('Word Count per Message', fontweight='bold')
axes[2].set_ylabel('Word Count')

plt.tight_layout()
plt.savefig('../outputs/step1_exploration.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to outputs/step1_exploration.png")