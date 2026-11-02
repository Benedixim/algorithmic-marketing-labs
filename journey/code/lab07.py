"""Lab 7. Product popularity ranking, factors and a profile to test.

Ranking by household penetration is allowed by the assignment. These are not
star ratings. Compare fixed grocery products; predict next-period penetration
with pre-period information and a product holdout.
"""
from common import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

t=transactions();p=table('product').set_index('PRODUCT_ID')
grocery=set(p[p.DEPARTMENT.eq('GROCERY')].index)
d=t[t.PRODUCT_ID.isin(grocery)]
rank=d.groupby('PRODUCT_ID').agg(households=('household_key','nunique'),baskets=('BASKET_ID','nunique'),
    paid=('paid','sum'),price=('unit_price','median')).sort_values(['households','baskets'],ascending=False)
rank=rank.join(p[['COMMODITY_DESC','BRAND','CURR_SIZE_OF_PRODUCT']])
rank['penetration']=rank.households/t.household_key.nunique()
pre=d[d.DAY<=530].groupby('PRODUCT_ID').agg(prior_households=('household_key','nunique'),
    price=('unit_price','median'),stores=('STORE_ID','nunique'),discount_share=('discounted','mean'),rows=('DAY','size'))
future=d[d.DAY>530].groupby('PRODUCT_ID').household_key.nunique()
features=pre[pre.rows>=30].join(future.rename('target')).fillna({'target':0})
X=features[['prior_households','price','stores','discount_share']]
tr,te=train_test_split(features.index,test_size=.25,random_state=42)
model=RandomForestRegressor(n_estimators=200,min_samples_leaf=10,random_state=42,n_jobs=1)
model.fit(X.loc[tr],features.loc[tr,'target']);pred=model.predict(X.loc[te])
baseline=features.loc[tr,'target'].median()
prior_window=d[d.DAY.between(350,530)].groupby('PRODUCT_ID').household_key.nunique()
prior_prediction=prior_window.reindex(te).fillna(0)
top=rank.head(5).reset_index().to_dict('records')
for r in top:r['category_ru']=category_ru(r['COMMODITY_DESC'])
rank.head(100).to_csv(OUT/'lab07_top_products.csv')
save('lab07',dict(products=len(rank),households=int(t.household_key.nunique()),top=top,
    feature_products=len(features),train=len(tr),test=len(te),cutoff=530,
    model_mae=float(mean_absolute_error(features.loc[te,'target'],pred)),
    baseline_mae=float(mean_absolute_error(features.loc[te,'target'],prior_prediction)),
    median_mae=float(mean_absolute_error(features.loc[te,'target'],np.repeat(baseline,len(te)))),
    importances=dict(zip(X.columns,map(float,model.feature_importances_))),
    profile=rank.head(20).agg({'price':'median','households':'median'}).to_dict(),
    definition='Distinct purchasing households per product over the study period; not customer star ratings.',
    recommendation='Prioritise availability of high-penetration products; test distribution and promotion changes. Feature importance indicates predictive association, not causality.'))
print('Lab 7',top[0]['category_ru'],top[0]['households'],'households')
