"""Lab 9. Actual buyer profiles, communications strategy and LLM invocation.
python code/lab09.py --prepare only saves inputs; otherwise runs the model.
"""
from common import *
import subprocess,sys,os

d=rfm();t=transactions();snapshot=int(t.DAY.max())+1
t=t[(t.DAY>=snapshot-365)&(t.DAY<snapshot)].merge(table('product')[['PRODUCT_ID','COMMODITY_DESC']],on='PRODUCT_ID',validate='many_to_one')
spec={
 'Лучшие':('Recent frequent high-spend shoppers','Thank frequent shoppers and invite them to plan their next grocery trip','Поблагодарить за регулярные покупки и помочь собрать следующую корзину'),
 'Под риском':('Previously frequent or high-spend shoppers with a long gap','Invite a return visit without pressure or an invented offer','Напомнить о привычных покупках и предложить вернуться без давления'),
 'Неактивные':('Shoppers with a long gap and modest previous activity','Offer a simple invitation to browse grocery products again','Коротко напомнить об ассортименте и пригласить посмотреть товары'),
 'Недавние редкие':('Recent shoppers with few grocery trips','Help plan the next grocery basket','Помочь спланировать следующую покупку'),
 'Остальные':('Shoppers with medium observed recency and frequency','Invite shoppers to refill their usual groceries','Напомнить о пополнении привычных продуктов')}
inputs=[]
for name,(audience,goal,strategy) in spec.items():
    g=d[d.segment==name]
    if g.empty:continue
    rows=t[t.household_key.isin(g.index)]
    category=rows.groupby('COMMODITY_DESC').household_key.nunique().idxmax()
    inputs.append(dict(segment=name,n=len(g),R=float(g.R.median()),F=float(g.F.median()),M=float(g.M.median()),
        category=category,category_ru=category_ru(category),audience=audience,goal=goal,strategy=strategy))
save('lab09_inputs',inputs)
if '--prepare' not in sys.argv:
    subprocess.run([os.environ.get('MARKETING_NODE','node'),str(ROOT/'code/generate.mjs')],check=True)
