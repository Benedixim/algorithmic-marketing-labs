"""Lab 1 points 1–4. Audit, descriptive analysis, shopper clustering."""
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hashlib

names = ['transaction_data','product','hh_demographic','campaign_desc','campaign_table','coupon','coupon_redempt']
audit = []
for name in names:
    d = table(name)
    missing=d.isna().sum().astype(int).to_dict()
    for c in d.select_dtypes(include='object'):
        missing[c] += int(d[c].fillna('x').str.strip().eq('').sum())
    audit.append(dict(table=name, rows=len(d), columns=len(d.columns),
        duplicates=int(d.duplicated().sum()), missing=missing,
        types=d.dtypes.astype(str).to_dict(),
        sha256=hashlib.sha256((ROOT/'data'/(name+'.csv')).read_bytes()).hexdigest()))
t = transactions(); b = baskets(); end = int(t.DAY.max())+1
h = t[t.DAY >= end-90]
x = h.groupby('household_key').agg(baskets=('BASKET_ID','nunique'),
    paid=('paid','sum'), products=('PRODUCT_ID','nunique'))
z = StandardScaler().fit_transform(np.log1p(x))
labels = KMeans(n_clusters=3, n_init=20, random_state=42).fit_predict(z)
x['cluster'] = labels
summary = x.groupby('cluster').agg(n=('baskets','size'), baskets=('baskets','median'),
    paid=('paid','median'), products=('products','median')).sort_values('baskets')
summary['name'] = ['Редкие покупки','Обычная активность','Частые покупки']
weekly=t.groupby('WEEK_NO').agg(paid=('paid','sum'), households=('household_key','nunique'))
weekly.to_csv(OUT/'lab01_weekly.csv')
x.to_csv(OUT/'lab01_clusters.csv')
save('lab01', dict(audit=audit, raw_rows=len(table('transaction_data')), clean_rows=len(t),
    excluded_rows=len(table('transaction_data'))-len(t), households=int(t.household_key.nunique()),
    baskets=len(b), products=int(t.PRODUCT_ID.nunique()), stores=int(t.STORE_ID.nunique()),
    day_min=int(t.DAY.min()), day_max=int(t.DAY.max()), demographics=len(table('hh_demographic')),
    missing_demographics=int(t.household_key.nunique()-len(table('hh_demographic'))),
    amount=float(t.paid.sum()), basket_median=float(b.paid.median()),
    basket_units_spend_correlation=float(b[['paid','units']].corr().iloc[0,1]),
    clusters=summary.reset_index().to_dict('records'),
    silhouette=float(silhouette_score(z, labels, sample_size=min(len(x),2000),random_state=42)),
    cluster_window=[end-90,end-1], cluster_households=len(x)))
print('Lab 1 complete',len(t),'lines',t.household_key.nunique(),'households')
