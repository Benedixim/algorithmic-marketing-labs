"""Lab 6. One fixed product/store, observed prices and future price prediction.

Quantity zero is retained in the weekly panel on no-purchase weeks, but an
unobserved price is never forward-filled. Price-demand association is not causal
elasticity; no profit optimum is asserted without costs or a price experiment.
"""
from common import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

t=transactions();p=table('product').set_index('PRODUCT_ID')
# Select by coverage, not by the sign or significance of the final association.
fixed_pack=p[(p.COMMODITY_DESC=='FLUID MILK PRODUCTS') & p.CURR_SIZE_OF_PRODUCT.str.strip().eq('1 GA')].index
g=t[t.PRODUCT_ID.isin(fixed_pack)].groupby(['PRODUCT_ID','STORE_ID']).agg(n=('DAY','size'),weeks=('WEEK_NO','nunique'),levels=('unit_price','nunique'))
g=g[(g.weeks>=60)&(g.levels>=5)].sort_values(['weeks','n'],ascending=False)
if g.empty:raise ValueError('No sufficiently observed product/store')
product,store=map(int,g.index[0]);x=t[(t.PRODUCT_ID==product)&(t.STORE_ID==store)]
w=x.groupby('WEEK_NO').agg(units=('QUANTITY','sum'),paid=('paid','sum'),price=('unit_price','median'),
    transactions=('BASKET_ID','nunique'),shelf=('shelf_price','median'))
w=w.reindex(range(int(t.WEEK_NO.min()),int(t.WEEK_NO.max())+1))
w['units']=w.units.fillna(0); w['lag_price']=w.price.shift(1);w['week']=w.index
w['sin']=np.sin(2*np.pi*w.index/52);w['cos']=np.cos(2*np.pi*w.index/52)
observed=w.dropna(subset=['price']); model_data=w.dropna(subset=['price','lag_price'])
cut=int(np.quantile(model_data.index,.8));tr=model_data[model_data.index<=cut];te=model_data[model_data.index>cut]
features=['lag_price','week','sin','cos']
model=RandomForestRegressor(n_estimators=200,min_samples_leaf=4,random_state=42,n_jobs=1)
model.fit(tr[features],tr.price);pred=model.predict(te[features])
olsdata=observed[(observed.units>0)&(observed.price>0)].copy()
olsdata['logprice']=np.log(olsdata.price);olsdata['logunits']=np.log(olsdata.units)
reg=LinearRegression().fit(olsdata[['logprice','week','sin','cos']],olsdata.logunits)
lower=float(observed.price.quantile(.25));upper=float(observed.price.quantile(.75))
bands=[]
for name,mask in [('Низкая цена',observed.price<=lower),('Высокая цена',observed.price>=upper)]:
    z=observed[mask];bands.append(dict(name=name,weeks=len(z),price=float(z.price.median()),units=float(z.units.mean()),revenue=float(z.paid.mean())))
w.to_csv(OUT/'lab06_weekly.csv',index_label='week_number')
meta=p.loc[product]
save('lab06',dict(product=product,store=store,category=meta.COMMODITY_DESC,category_ru=category_ru(meta.COMMODITY_DESC),
    subcategory=meta.SUB_COMMODITY_DESC,package=str(meta.CURR_SIZE_OF_PRODUCT),brand=meta.BRAND,
    raw_rows=len(x),observed_weeks=len(observed),total_weeks=len(w),missing_price_weeks=int(w.price.isna().sum()),
    price_min=float(observed.price.min()),price_max=float(observed.price.max()),
    correlation=float(observed[['price','units']].corr().iloc[0,1]),
    association_coef=float(reg.coef_[0]),
    train=len(tr),test=len(te),test_start=int(te.index.min()),
    model_mae=float(mean_absolute_error(te.price,pred)),baseline_mae=float(mean_absolute_error(te.price,te.lag_price)),
    bands=bands,recommendation='Run a small randomised price experiment around observed prices with stock and unit margin controls. Do not set an estimated profit optimum from observational association alone.'))
print('Lab 6',product,store,meta.COMMODITY_DESC,'weeks',len(observed),'MAE',mean_absolute_error(te.price,pred))
