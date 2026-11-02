"""Пункты 1–4. Общий бизнес-контекст, аудит, аналитика и решение задачи группировки."""
from common import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

names=['orders','customers','order_items','order_payments','products','sellers','order_reviews','marketing_qualified_leads','closed_deals','translation']
audit={n:dict(rows=len(table(n)),columns=len(table(n).columns),duplicates=int(table(n).duplicated().sum()),dtypes={c:str(v) for c,v in table(n).dtypes.items()},missing=table(n).isna().sum().to_dict()) for n in names}
o=delivered(); d=items(); t=customer_rfm()
x=StandardScaler().fit_transform(np.log1p(t[['R','F','M']]))
t['cluster']=KMeans(n_clusters=3,n_init=20,random_state=42).fit_predict(x)
monthly=o.groupby(o.order_purchase_timestamp.dt.to_period('M').astype(str)).agg(orders=('order_id','size'),value=('value','sum'))
cats=d.groupby('category').agg(rows=('price','size'),value=('price','sum')).sort_values('value',ascending=False)
t.to_csv(OUT/'lab01_customers.csv')
monthly.to_csv(OUT/'lab01_monthly.csv');cats.to_csv(OUT/'lab01_categories.csv')
save('lab01',dict(audit=audit,period=[str(table('orders').order_purchase_timestamp.min()),str(table('orders').order_purchase_timestamp.max())],analysis_period=['2017-01-01','2018-08-31'],orders=len(o),lines=len(d),customers=t.shape[0],products=d.product_id.nunique(),sellers=d.seller_id.nunique(),value=o.value.sum(),median_order=o.value.median(),q_order=o.value.quantile([.25,.5,.75,.95]).to_dict(),top_categories=cats.head(6).reset_index().to_dict('records'),rfm_correlation=t[['R','F','M']].corr().round(3).to_dict(),clusters=t.groupby('cluster').agg(n=('R','size'),R=('R','median'),F=('F','median'),M=('M','median')).round(2).reset_index().to_dict('records'),silhouette=silhouette_score(x,t.cluster,sample_size=3000,random_state=42)))
