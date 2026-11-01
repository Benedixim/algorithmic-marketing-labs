"""ЛР 9. Портреты RFM-сегментов и локальная генерация рекламного текста.
Подготовка: python lab09.py --prepare
Генерация: npm install (в корне проекта), затем python lab09.py
Node.js выполняет открытую LLM через Transformers.js. Секреты и API не нужны.
"""
from common import *
import sys
import subprocess
import os

df=retail();t=segments(df)
specs=[('Лояльные','frequent returning gift-shop customers','show an item for their next order'),
       ('Новые','customers who recently placed their first orders','encourage them to browse another gift'),
       ('Под риском','previously active customers who have not ordered recently','invite them to visit the gift collection again'),
       ('Спящие','customers with an old order and little purchase history','gently invite them to browse the shop again without pressure'),
       ('Остальные','occasionally returning customers','suggest an item to browse on their next visit')]
inputs=[]
for seg,audience,goal in specs:
    ids=t.index[t.segment==seg]
    purchases=df[df.CustomerID.isin(ids)]
    popular=purchases.groupby('StockCode').InvoiceNo.nunique().idxmax()
    name=purchases.loc[purchases.StockCode==popular,'Description'].mode().iloc[0].strip()
    profile=t.loc[ids]
    inputs.append(dict(segment=seg,customers=len(ids),median_recency=int(profile.R.median()),median_orders=float(profile.F.median()),median_spend=float(profile.M.median()),product=name,audience=audience,goal=goal))
save('lab09_inputs',inputs)
if '--prepare' not in sys.argv:
    node=os.environ.get('MARKETING_NODE','node')
    subprocess.run([node,str(ROOT/'code'/'generate.mjs')],check=True)
