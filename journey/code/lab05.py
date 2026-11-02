"""Lab 5, assignment points 1–3. Real electronics reviews from Amazon.

Use only the authors' positive/negative files for classification. Their balanced
sample does not estimate the share of dissatisfied Amazon customers. Stars are
proxy labels for sentiment; the model sees review_text only. Product identifiers
and duplicate texts never cross from training into the holdout.
"""
from sentiment_common import reviews, save, RESULTS
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.dummy import DummyClassifier

raw = reviews()
electronics = raw[(raw.domain == 'electronics') & raw.subset.isin(['positive', 'negative'])].copy()
data = electronics[electronics.norm.ne('') & electronics.asin.ne('') &
                   electronics.rating.isin([1, 2, 4, 5])].copy()
data = data.sort_values(['date', 'asin']).drop_duplicates('norm').reset_index(drop=True)
data['positive'] = (data.rating >= 4).astype(int)
train_i, test_i = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                       .split(data, data.positive, groups=data.asin))
train, test = data.iloc[train_i], data.iloc[test_i]
assert set(train.asin).isdisjoint(test.asin)
assert set(train.norm).isdisjoint(test.norm)
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True),
    LogisticRegression(max_iter=1500, random_state=42))
model.fit(train.review_text, train.positive)
predictions = model.predict(test.review_text)
baseline = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train), 1)), train.positive)
baseline_predictions = baseline.predict(np.zeros((len(test), 1)))

# Transparent keyword themes, not an automatic attribution of root causes.
# Several themes can occur in one review; a mention can be negated or incidental.
themes = {
    'Поломки и отказ в работе': r'\b(broke|broken|defect\w*|fail\w*|stopped working|doesn.t work|didn.t work|not work\w*)\b',
    'Качество звука': r'\b(sound|audio|volume|noise|speaker\w*|headphone\w*)\b',
    'Установка и подключение': r'\b(install\w*|setup|set.up|driver\w*|connect\w*|compatib\w*)\b',
    'Батарея и питание': r'\b(batter\w*|charg\w*|power|adapter\w*)\b',
    'Поддержка и возврат': r'\b(customer service|support|refund\w*|return\w*|warrant\w*)\b',
}
negative = data[data.positive == 0].copy()
topic_results = []
for name, pattern in themes.items():
    mask = negative.norm.str.contains(pattern.replace('(', '(?:'), regex=True)
    topic_results.append({'theme': name, 'count': int(mask.sum()),
                          'share': float(mask.mean()), 'pattern': pattern})
topic_results.sort(key=lambda r: r['count'], reverse=True)

vocabulary = model[0].get_feature_names_out()
weights = model[1].coef_[0]
features = {'positive': vocabulary[np.argsort(weights)[-12:][::-1]].tolist(),
            'negative': vocabulary[np.argsort(weights)[:12]].tolist()}
test_output = test[['asin', 'rating', 'positive']].copy()
test_output['predicted_positive'] = predictions
test_output.to_csv(RESULTS / 'lab05_holdout.csv', index=False)
save('lab05', {
    'archive_records': len(raw), 'raw_electronics_labelled': len(electronics),
    'raw_fields': 11, 'retained_fields': 7,
    'missing_labelled': {k: int(electronics[k].isna().sum() + electronics[k].eq('').sum())
                        for k in ['asin', 'review_text', 'rating', 'date']},
    'duplicate_texts_removed': int(len(electronics) - len(data)),
    'analysed': len(data), 'products': int(data.asin.nunique()),
    'start': str(data.date.min().date()), 'end': str(data.date.max().date()),
    'train': len(train), 'test': len(test),
    'train_products': int(train.asin.nunique()), 'test_products': int(test.asin.nunique()),
    'product_overlap': 0, 'text_overlap': 0,
    'test_positive': int(test.positive.sum()), 'test_negative': int((test.positive == 0).sum()),
    'accuracy': float(accuracy_score(test.positive, predictions)),
    'macro_f1': float(f1_score(test.positive, predictions, average='macro')),
    'baseline_accuracy': float(accuracy_score(test.positive, baseline_predictions)),
    'confusion': confusion_matrix(test.positive, predictions, labels=[0, 1]).tolist(),
    'negative': len(negative), 'positive': int(data.positive.sum()),
    'stars': {str(int(k)): int(v) for k, v in data.rating.value_counts().sort_index().items()},
    'topics': topic_results, 'model_features': features,
    'label_rule': '1–2 stars negative; 4–5 positive. Labels describe star agreement, not independent expert sentiment labels.',
    'split_rule': 'GroupShuffleSplit by product ASIN, random_state 42, 20% of product groups in holdout.',
    'recommendations': [
        'Проверять упоминания отказов вручную и передавать повторяющиеся случаи в контроль качества.',
        'Дополнить карточки совместимостью и понятной инструкцией по подключению.',
        'Показать понятный порядок обращения за поддержкой и возврата.',
    ],
})
print('Lab 5 complete', len(data), 'reviews;', len(test), 'holdout;',
      round(accuracy_score(test.positive, predictions), 4), 'accuracy')
