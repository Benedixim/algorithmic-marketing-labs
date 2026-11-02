"""Пункты 2–4. RFM с явными правилами и адресными действиями.
R — Recency, F — Frequency, M — Monetary. День среза 1 сентября 2018.
"""
from common import *
t=customer_rfm();t['priority']=t.M.rank(pct=True)
g=t.groupby('segment').agg(n=('R','size'),R=('R','median'),F=('F','median'),M=('M','median'),value=('M','sum'))
g['customer_share']=g.n/len(t);g['value_share']=g.value/t.M.sum()
t.to_csv(OUT/'lab08_customers.csv');g.to_csv(OUT/'lab08_segments.csv')
save('lab08',dict(date=str(SNAPSHOT.date()),n=len(t),repeat=int((t.F>=2).sum()),high_spend=float(t.M.quantile(.75)),segments=g.round(4).reset_index().to_dict('records'),rules={'Повторные':'R ≤ 90 дней и F ≥ 2','Новые':'R ≤ 30 дней и F = 1','Под риском':'R > 90 дней и (F ≥ 2 или M не ниже верхнего квартиля)','Неактивные':'R > 90 дней и остальные условия не выполнены','Разовые':'30 < R ≤ 90 дней и F = 1'},monetary_rule='В группе риска учитываем также клиентов из верхней четверти по общей сумме покупок M. Внутри групп приоритет контакта задаём по M.'))
