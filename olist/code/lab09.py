"""Пункты 2–5. Портреты, коммуникационная стратегия и запуск генеративной модели.
python code/lab09.py --prepare — только входы. Без флага также запускает модель.
"""
from common import *
import sys, subprocess, os
t=customer_rfm();d=items()
specs={
 'Повторные':('Покупали несколько раз и вернулись за последние 90 дней','Предложить следующую покупку в знакомой категории','Показать подборку для следующего заказа'),
 'Новые':('Сделали один заказ в последние 30 дней','Помочь выбрать товар для второго заказа','Пригласить посмотреть ещё одну категорию'),
 'Под риском':('Покупали повторно или потратили крупную сумму, но не возвращались более 90 дней','Напомнить о магазине без давления','Пригласить вернуться к просмотру каталога'),
 'Неактивные':('Сделали одну покупку и не возвращались более 90 дней','Напомнить о возможности посмотреть каталог','Мягкое приглашение без давления'),
 'Разовые':('Сделали одну покупку от 31 до 90 дней назад','Предложить следующую покупку','Предложить посмотреть популярную категорию')}
inputs=[]
for seg,(audience,goal,strategy) in specs.items():
    p=t[t.segment==seg];purchases=d[d.customer_unique_id.isin(p.index)]
    category=purchases.groupby('category').order_id.nunique().idxmax()
    inputs.append(dict(segment=seg,n=len(p),R=float(p.R.median()),F=float(p.F.median()),M=float(p.M.median()),category=category,category_ru=category_ru(category),audience=audience,goal=goal,strategy=strategy))
save('lab09_inputs',inputs)
if '--prepare' not in sys.argv:
    subprocess.run([os.environ.get('MARKETING_NODE','node'),str(ROOT/'code'/'generate.mjs')],check=True)
