"""Пункты 1–4. Отток продавцов как клиентов платформы Olist.
Отток операционно определён как отсутствие заказов следующие 60 дней.
Обучение, выбор порога и тест разделены по времени, признаки только из прошлого.
"""
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score

# Статус окончательной доставки не фильтруем. В момент среза известно,
# что заказ с товарными позициями уже оформлен, а не его будущий итог.
d=table('order_items').merge(table('orders')[['order_id','order_purchase_timestamp']],on='order_id',validate='many_to_one')
d=d.merge(table('products')[['product_id','product_category_name']],on='product_id',how='left',validate='many_to_one')
d['category']=d.product_category_name.fillna('unknown')
def snapshot(date):
    date=pd.Timestamp(date)
    past=d[(d.order_purchase_timestamp<date)&(d.order_purchase_timestamp>=date-pd.Timedelta(days=90))]
    t=past.groupby('seller_id').agg(orders90=('order_id','nunique'),value90=('price','sum'),last=('order_purchase_timestamp','max'),categories=('category','nunique'),avg_price=('price','mean'))
    t=t[t.orders90>=3].copy()
    t['recency']=(date-t['last']).dt.total_seconds()/86400
    recent=past[past.order_purchase_timestamp>=date-pd.Timedelta(days=30)].groupby('seller_id').order_id.nunique()
    t['orders30']=recent.reindex(t.index).fillna(0)
    t['recent_share']=t.orders30/t.orders90
    future=d[(d.order_purchase_timestamp>=date)&(d.order_purchase_timestamp<date+pd.Timedelta(days=60))]
    t['inactive60']=(~t.index.isin(future.seller_id)).astype(int)
    # Ни факт завершения доставки, ни будущая оценка не используются в признаках.
    return t
tr=snapshot('2017-12-01');va=snapshot('2018-03-01');te=snapshot('2018-06-01')
features=['orders90','value90','categories','avg_price','recency','orders30','recent_share']
def x(t):
    v=t[features].copy()
    for c in ['orders90','value90','avg_price','orders30']:v[c]=np.log1p(v[c])
    return v
model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))
model.fit(x(tr),tr.inactive60)
vp=model.predict_proba(x(va))[:,1]
threshold=float(max(np.arange(.05,.91,.01),key=lambda v:f1_score(va.inactive60,vp>=v)))
prob=model.predict_proba(x(te))[:,1];yp=prob>=threshold;y=te.inactive60
count=int(np.ceil(len(te)*.2));selected=np.argsort(-prob)[:count]
te['risk']=prob;te.to_csv(OUT/'lab10_sellers.csv')
by_recent=te.assign(group=np.where(te.recency>30,'Более 30 дней без заказов','До 30 дней без заказов')).groupby('group').agg(n=('inactive60','size'),rate=('inactive60','mean'))
save('lab10',dict(train=len(tr),validation=len(va),test=len(te),test_date='2018-06-01',horizon_days=60,train_rate=tr.inactive60.mean(),validation_rate=va.inactive60.mean(),test_rate=y.mean(),threshold=threshold,auc=roc_auc_score(y,prob),average_precision=average_precision_score(y,prob),precision=precision_score(y,yp,zero_division=0),recall=recall_score(y,yp,zero_division=0),f1=f1_score(y,yp),top_n=count,top_rate=y.iloc[selected].mean(),top_actual=int(y.iloc[selected].sum()),coefficients=dict(zip(features,model[-1].coef_[0])),recency_groups=by_recent.reset_index().to_dict('records'),note='Нет записи о расторжении договора. Измеряем прекращение продаж на 60 дней. Когорту ограничиваем продавцами с минимум тремя заказами за прошлые 90 дней.'))
