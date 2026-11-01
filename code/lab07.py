"""ЛР 7. Рейтинг товаров по повторной покупке за 60 дней (Wilson)."""
from common import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score

df=retail()
events=df[['CustomerID','StockCode','InvoiceDate']].copy()
events['date']=events.InvoiceDate.dt.normalize()
events=events.drop_duplicates(['CustomerID','StockCode','date']).sort_values('date')
first=events.groupby(['CustomerID','StockCode']).date.min().rename('first')
e=events.join(first,on=['CustomerID','StockCode'])
e['lag']=(e.date-e['first']).dt.days
repeat=e[(e.lag>0)&(e.lag<=60)].groupby(['CustomerID','StockCode']).size().gt(0).rename('repeat')
last=df.InvoiceDate.max().normalize()
buyers=first[first<=last-pd.Timedelta(days=60)].to_frame().join(repeat).fillna({'repeat':False})
rank=buyers.groupby('StockCode').agg(n=('repeat','size'),k=('repeat','sum'))
rank=rank[rank.n>=100].copy()
z=1.96;p=rank.k/rank.n;n=rank.n
rank['rate']=p
rank['score']=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
names=df.groupby('StockCode').Description.agg(lambda x:x.mode().iloc[0])
rank=rank.join(names.rename('name')).join(df.groupby('StockCode').UnitPrice.median().rename('median_price')).sort_values('score',ascending=False)
# Описательная модель доли повторов на отложенных товарах.
# Признаки относятся к тому же периоду: это не временной прогноз.
features=rank[['median_price','n']].copy()
features['median_quantity']=df.groupby('StockCode').Quantity.median().reindex(rank.index)
features['countries']=df.groupby('StockCode').Country.nunique().reindex(rank.index)
features=np.log1p(features)
tr,te=train_test_split(np.arange(len(rank)),test_size=.2,random_state=42)
model=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(features.iloc[tr],rank.rate.iloc[tr])
pred=np.clip(model.predict(features.iloc[te]),0,1)
baseline=np.repeat(rank.rate.iloc[tr].mean(),len(te))
prediction=dict(test_products=len(te),mae=mean_absolute_error(rank.rate.iloc[te],pred),baseline_mae=mean_absolute_error(rank.rate.iloc[te],baseline),r2=r2_score(rank.rate.iloc[te],pred),features=list(features.columns),note='Случайно отложенные товары, признаки и рейтинг за один период. Проверка описательной зависимости, не будущего спроса.')
rank.to_csv(OUT/'lab07_ranking.csv')
save('lab07',dict(n_products=len(rank),eligible_pairs=len(buyers),window_days=60,min_buyers=100,missing=int(rank.isna().sum().sum()),top=rank.reset_index().head(5).to_dict('records'),corr_price_rate=float(rank[['median_price','rate']].corr().iloc[0,1]),corr_buyers_rate=float(rank[['n','rate']].corr().iloc[0,1]),prediction=prediction,note='В наборе нет звёзд и оценок. Рейтинг отражает повторные покупки, а не удовлетворённость. Wilson учитывает размер выборки. Корреляции не доказывают влияние характеристик товара.'))
