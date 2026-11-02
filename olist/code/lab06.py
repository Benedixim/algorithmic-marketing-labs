"""Пункты 1–4. Цена одного товара одного продавца.
Критерии выбора заданы до моделирования. Сравнения являются наблюдательными.
"""
from common import *
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

d=items()
g=d.groupby(['seller_id','product_id']).agg(n=('price','size'),prices=('price','nunique'),first=('order_purchase_timestamp','min'),last=('order_purchase_timestamp','max'))
g['span']=(g['last']-g['first']).dt.days
eligible=g[(g.n>=100)&(g.prices>=3)&(g.span>=180)].sort_values(['n'],ascending=False)
assert len(eligible)
seller,product=eligible.index[0]
a=d[(d.seller_id==seller)&(d.product_id==product)].copy()
a['date']=a.order_purchase_timestamp.dt.normalize()
day=a.groupby('date').agg(price=('price','median'),quantity=('price','size'),freight=('freight_value','median'))
day['t']=(day.index-day.index.min()).days
day['month']=day.index.month;day['dow']=day.index.dayofweek
x=pd.concat([day[['t','freight']],pd.get_dummies(day[['month','dow']].astype(str),dtype=float)],axis=1)
split=int(len(day)*.8)
model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(x.iloc[:split],day.price.iloc[:split])
pred=model.predict(x.iloc[split:]);base=np.repeat(day.price.iloc[:split].median(),len(pred))
day.to_csv(OUT/'lab06_prices.csv')
levels=a.groupby('price').agg(lines=('price','size'),days=('date','nunique')).sort_values('lines',ascending=False)
levels['units_per_selling_day']=levels.lines/levels.days
save('lab06',dict(product=product,seller=seller,category=a.category.mode().iloc[0],category_ru=category_ru(a.category.mode().iloc[0]),selection=eligible.head(5).reset_index().assign(first=lambda x:x['first'].astype(str),last=lambda x:x['last'].astype(str)).to_dict('records'),lines=len(a),days=len(day),span=int((day.index.max()-day.index.min()).days),min_price=a.price.min(),max_price=a.price.max(),median_price=a.price.median(),distinct_prices=a.price.nunique(),correlation=day[['price','quantity']].corr().iloc[0,1],levels=levels.head(8).reset_index().to_dict('records'),train_days=split,test_days=len(day)-split,mae=mean_absolute_error(day.price.iloc[split:],pred),baseline_mae=mean_absolute_error(day.price.iloc[split:],base),period=[str(day.index.min().date()),str(day.index.max().date())],test_start=str(day.index[split].date()),note='Модель предсказывает цену, а не причинный эффект цены на спрос. В данных нет затрат и остатков. Оптимальную прибыль и скидки не вычисляем без этих величин.'))
