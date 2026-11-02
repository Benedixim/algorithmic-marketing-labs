"""Пункты 1–3. Привлечение продавцов Olist и рекомендации по источникам.
Реклама ведёт на посадочную страницу подключения продавца.
Учитываем равное окно 60 дней после первой заявки, а не любое будущее закрытие.
В данных нет затрат, показов и креативов. Доходность рекламы не выдумываем.
"""
from common import *

m=table('marketing_qualified_leads').copy();w=table('closed_deals')
joined=m.merge(w[['mql_id','seller_id','won_date']],on='mql_id',how='left',validate='one_to_one')
joined['lag']=(joined.won_date-joined.first_contact_date).dt.total_seconds()/86400
joined['won60']=joined.lag.between(0,60)
# Все заявки до 1 июня имеют одинаковое 60-дневное окно до конца июля.
joined=joined[joined.first_contact_date<'2018-06-01'].copy()
joined['origin']=joined.origin.fillna('unknown')
g=joined.groupby('origin').agg(leads=('mql_id','size'),wins=('won60','sum'))
g['rate']=g.wins/g.leads
z=1.96;p=g.rate;n=g.leads
g['low']=(p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
g['high']=(p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
g=g.sort_values('leads',ascending=False)
linked=set(w.seller_id)&set(table('sellers').seller_id)
# При одинаковой ценности подключённого продавца стоимость подключения
# равна стоимости заявки, делённой на конверсию. Этот коэффициент
# показывает допустимую стоимость заявки относительно платного поиска.
reference=float(g.loc['paid_search','rate'])
cost_rule={'reference':'paid_search','assumption':'Одинаковая экономическая ценность подключённого продавца по каналам','max_lead_cost_ratio':{k:float(g.loc[k,'rate']/reference) for k in ['social','email','display']},'formula':'cost_per_connected_seller = cost_per_lead / conversion_60_days'}
g.to_csv(OUT/'lab04_channels.csv')
save('lab04',dict(raw_leads=len(m),raw_wins=len(w),linked_sellers=len(linked),missing_origin=int(m.origin.isna().sum()),period=[str(m.first_contact_date.min()),str(m.first_contact_date.max())],leads=len(joined),wins=int(joined.won60.sum()),rate=joined.won60.mean(),channels=g.reset_index().to_dict('records'),median_days=joined.loc[joined.won60,'lag'].median(),cost_rule=cost_rule,method='Конверсия заявки в подключение за 60 дней. Различия описательные, без причинных выводов и выдуманных рекламных затрат.'))
