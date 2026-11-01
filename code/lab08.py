"""ЛР 8. RFM на дату 10.12.2011, сегменты и адресные действия."""
from common import *

df=retail();t=segments(df)
summary=t.groupby('segment').agg(customers=('R','size'),revenue=('M','sum'),R=('R','median'),F=('F','median'),M=('M','median')).reset_index()
summary['share']=summary.customers/len(t)
summary['revenue_share']=summary.revenue/summary.revenue.sum()
t.to_csv(OUT/'lab08_customers.csv')
save('lab08',dict(date='2011-12-10',customers=len(t),summary=summary.round(4).to_dict('records'),score_distribution=t[['Rs','Fs','Ms']].apply(lambda a:a.value_counts()).fillna(0).to_dict(),rules={'Лояльные':'Rs >= 4, Fs >= 4, Ms >= 4','Под риском':'Rs <= 2, Fs >= 3','Новые':'Rs >= 4, F <= 2, первая наблюдаемая покупка за последние 60 дней','Спящие':'Rs <= 2; кроме под риском','Остальные':'все остальные'},note='F — число уникальных заказов, M — сумма товарных покупок в GBP. Возвраты исключены, поэтому M не является чистой выручкой.'))
