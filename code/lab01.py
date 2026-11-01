"""ЛР 1. Знакомство с данными и сегментация покупателей KMeans."""
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

raw = retail(raw=True)
df = retail()
t = rfm(df)
x = StandardScaler().fit_transform(np.log1p(t))
model = KMeans(n_clusters=4, random_state=42, n_init=10)
t['cluster'] = model.fit_predict(x)
monthly = df.assign(month=df.InvoiceDate.dt.to_period('M').astype(str), revenue=df.Quantity*df.UnitPrice).groupby('month').revenue.sum()
t.to_csv(OUT/'lab01_clusters.csv')
save('lab01', dict(raw_rows=len(raw), columns=len(raw.columns), period=[str(raw.InvoiceDate.min()),str(raw.InvoiceDate.max())],
    missing=raw.isna().sum().to_dict(), duplicates=int(raw.duplicated().sum()), clean_rows=len(df),
    customers=df.CustomerID.nunique(), products=df.StockCode.nunique(), invoices=df.InvoiceNo.nunique(),
    revenue=float((df.Quantity*df.UnitPrice).sum()), monthly=monthly.to_dict(),
    silhouette=silhouette_score(x,t.cluster,sample_size=2000,random_state=42),
    clusters=t.groupby('cluster').agg(n=('R','size'),R=('R','median'),F=('F','median'),M=('M','median')).round(2).reset_index().to_dict('records'),
    correlation=t[['R','F','M']].corr().round(3).to_dict()))
