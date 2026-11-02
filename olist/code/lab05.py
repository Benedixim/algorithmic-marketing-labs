"""Пункты 1–3. Португалоязычные отзывы покупателей Olist.
Звёзды дают приближённую метку тональности, не экспертную разметку текста.
"""
from common import *
import re, unicodedata
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

r=table('order_reviews').copy()
r['text']=r.review_comment_message.fillna('').str.strip()
written=r[r.text.ne('')].copy()
written['norm']=written.text.str.lower().str.replace(r'\s+',' ',regex=True)
written=written.sort_values('review_answer_timestamp').drop_duplicates('norm')
d=written[written.review_score.ne(3)].copy();d['positive']=(d.review_score>=4).astype(int)
tr=d[d.review_answer_timestamp<'2018-05-01'];te=d[d.review_answer_timestamp>='2018-05-01']
assert set(tr.norm).isdisjoint(te.norm)
model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=30000,sublinear_tf=True),LogisticRegression(max_iter=1500,random_state=42))
model.fit(tr.text,tr.positive);pred=model.predict(te.text)
negative=written[written.review_score<=2].copy()
negative['plain']=negative.text.map(lambda s:''.join(c for c in unicodedata.normalize('NFKD',s.lower()) if not unicodedata.combining(c)))
topics={'Доставка':r'atras|demor|prazo|entreg','Повреждения и дефекты':r'defeit|quebrad|danific|avari','Другой товар':r'errad|diferent|troc','Неполный заказ':r'falt|incomplet'}
counts={k:int(negative.plain.str.contains(v,regex=True).sum()) for k,v in topics.items()}
save('lab05',dict(raw=len(r),text_rows=int(r.text.ne('').sum()),missing_text=int(r.text.eq('').sum()),distinct_texts=len(written),negative=int((written.review_score<=2).sum()),positive=int((written.review_score>=4).sum()),neutral=int((written.review_score==3).sum()),train=len(tr),test=len(te),accuracy=accuracy_score(te.positive,pred),f1=f1_score(te.positive,pred),confusion=confusion_matrix(te.positive,pred).tolist(),topics=counts,stars=r.review_score.value_counts().sort_index().to_dict(),language='Portuguese',label_rule='1–2 звезды отрицательные, 4–5 положительные. 3 звезды исключаем из обучения. Темы пересекаются.'))
