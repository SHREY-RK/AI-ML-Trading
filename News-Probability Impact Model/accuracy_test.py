import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from clean_text import clean_text

df = pd.read_csv('data/dataset.csv')
df['label'] = df['company'] + '::' + df['movement']
df['clean'] = df['text'].apply(clean_text)

# Split by unique news text so same news doesn't appear in both train and test
texts = df['text'].unique()
train_texts = texts[:int(len(texts)*0.8)]
test_texts = texts[int(len(texts)*0.8):]

train_df = df[df['text'].isin(train_texts)]
test_df = df[df['text'].isin(test_texts)]

model = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2))),
    ('clf', MultinomialNB())
])
model.fit(train_df['clean'], train_df['label'])

# For each test news, predict all company movements
results = []
for text in test_texts:
    clean = clean_text(text)
    rows = test_df[test_df['text'] == text]
    pred_label = model.predict([clean])[0]
    pred_company, pred_movement = pred_label.split('::')
    for _, row in rows.iterrows():
        actual_movement = row['movement']
        company = row['company']
        # Check if model correctly predicted this company's movement
        if company == pred_company:
            correct = pred_movement == actual_movement
        else:
            # Model didn't predict this company — check via probabilities
            probs = dict(zip(model.classes_, model.predict_proba([clean])[0]))
            best_label = max(
                [f"{company}::Up", f"{company}::Down", f"{company}::Neutral"],
                key=lambda l: probs.get(l, 0)
            )
            correct = best_label == f"{company}::{actual_movement}"
        results.append({'company': company, 'actual': actual_movement, 'correct': correct})

results_df = pd.DataFrame(results)
overall = results_df['correct'].mean() * 100
print(f'Overall Accuracy: {round(overall, 2)}%')
print(f'Random baseline: 33.33% (3 classes)')
print()
print('Per company accuracy:')
print(results_df.groupby('company')['correct'].mean().mul(100).round(2).to_string())
print()
print('Movement distribution in test set:')
print(results_df.groupby('actual').size().to_string())
