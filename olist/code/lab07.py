"""Пункты 1–4. Рейтинг продавцов по настоящим оценкам заказов.
Для однозначной привязки оценки оставляем заказы с одним продавцом.
"""
from common import *
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

o=delivered();one=o[o.sellers==1].copy()
seller=table('order_items')[['order_id','seller_id']].drop_duplicates('order_id')
r=table('order_reviews').sort_values('review_answer_timestamp').drop_duplicates('order_id',keep='last')
a=one.merge(seller,on='order_id',validate='one_to_one').merge(r[['order_id','review_score']],on='order_id',validate='one_to_one')
a=a.dropna(subset=['late_days','delivery_days']).copy();a['freight_share']=a.freight/(a.value+a.freight)
g=a.groupby('seller_id').agg(n=('review_score','size'),mean=('review_score','mean'))
g=g[g.n>=30].copy();prior=float(a.review_score.mean());strength=30
g['rating']=(g.n*g['mean']+strength*prior)/(g.n+strength)
g=g.sort_values('rating',ascending=False);g['seller_label']=[f'Продавец {i+1}' for i in range(len(g))]
features=['delivery_days','late_days','value','freight_share','items']
tr=a[a.order_purchase_timestamp<'2018-05-01'];te=a[a.order_purchase_timestamp>='2018-05-01']
model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=5,random_state=42))
model.fit(tr[features],(tr.review_score<=2).astype(int));y=(te.review_score<=2).astype(int)
prob=model.predict_proba(te[features])[:,1]
pi=permutation_importance(model,te[features],y,scoring='roc_auc',n_repeats=3,random_state=42)
g.to_csv(OUT/'lab07_ratings.csv')
timing=a.assign(late=a.late_days.gt(0)).groupby('late').agg(n=('review_score','size'),rating=('review_score','mean'),negative=('review_score',lambda x:(x<=2).mean()))
save('lab07',dict(orders=len(a),sellers=len(g),prior=prior,prior_weight=strength,top=g.head(5).reset_index().to_dict('records'),timing=timing.reset_index().to_dict('records'),correlations=a[['review_score','delivery_days','late_days','freight_share','value']].corr().review_score.to_dict(),train=len(tr),test=len(te),auc=roc_auc_score(y,prob),importance={k:float(v) for k,v in zip(features,pi.importances_mean)},note='Показываем связь характеристик выполненного заказа с оценкой. Данные о фактической доставке доступны после покупки. Это не прогноз оценки в момент оформления заказа.'))
