"""ЛР 6. Цены и дневной спрос на фиксированный SKU. Проверка по времени."""
from common import *
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

df=retail()
sku='84879' # ASSORTED COLOUR BIRD ORNAMENT, выбор фиксирован до обучения.
d=df[(df.StockCode==sku)&(df.Country=='United Kingdom')].copy()
d['date']=d.InvoiceDate.dt.normalize()
d['revenue']=d.Quantity*d.UnitPrice
daily=d.groupby('date').agg(q=('Quantity','sum'),revenue=('revenue','sum')).sort_index()
daily['price']=daily.revenue/daily.q
daily['t']=np.arange(len(daily))/len(daily)
daily['dow']=daily.index.dayofweek
daily['month']=daily.index.month
x=pd.DataFrame({'log_price':np.log(daily.price),'trend':daily.t},index=daily.index)
x=pd.concat([x,pd.get_dummies(daily[['dow','month']].astype(str),drop_first=True,dtype=float)],axis=1)
y=np.log1p(daily.q)
split=int(len(x)*.8)
model=Ridge(alpha=1).fit(x.iloc[:split],y.iloc[:split])
pred=np.maximum(0,np.expm1(model.predict(x.iloc[split:])))
baseline=np.repeat(daily.q.iloc[:split].median(),len(pred))
mae=mean_absolute_error(daily.q.iloc[split:],pred)
bm=mean_absolute_error(daily.q.iloc[split:],baseline)
# Сценарии в наблюдаемом диапазоне относительно медианы.
full=Ridge(alpha=1).fit(x,y)
p0=float(daily.price.median())
grid=np.linspace(max(float(d.UnitPrice.min()),p0*.9),min(float(d.UnitPrice.max()),p0*1.1),31)
scenarios=[]
template=x.tail(28).copy()
for p in grid:
    z=template.copy();z['log_price']=np.log(p)
    qty=float(np.maximum(0,np.expm1(full.predict(z))).mean())
    scenarios.append(dict(price=float(p),quantity=qty,revenue=float(p*qty)))
daily.to_csv(OUT/'lab06_daily.csv')
save('lab06',dict(sku=sku,name=d.Description.mode().iloc[0],rows=len(d),days=len(daily),missing=int(d[['UnitPrice','Quantity','InvoiceDate']].isna().sum().sum()),price_min=float(d.UnitPrice.min()),price_max=float(d.UnitPrice.max()),median_price=p0,distinct_prices=d.UnitPrice.nunique(),price_levels=d.groupby('UnitPrice').agg(rows=('Quantity','size'),median_quantity=('Quantity','median')).reset_index().to_dict('records'),correlation=float(daily[['price','q']].corr().iloc[0,1]),elasticity=float(full.coef_[0]),train_days=split,test_days=len(x)-split,mae=mae,baseline_mae=bm,scenarios=scenarios,best=max(scenarios,key=lambda r:r['revenue']),sample=daily.reset_index().assign(date=lambda a:a.date.astype(str)).to_dict('records'),note='Только дни с продажами. Нет остатков, трафика и цены конкурентов. Коэффициент не является причинной эластичностью, оптимум — модельный сценарий, для внедрения нужен ценовой эксперимент.'))
