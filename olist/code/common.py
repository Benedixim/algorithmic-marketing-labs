"""Общие таблицы одного кейса Olist. Все деньги указаны в бразильских реалах.

Два связанных публичных выпуска Olist описаны в sources.json.
Связываем строки товаров и заказы только many_to_one, чтобы не удвоить суммы.
"""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT/'data', ROOT/'results'
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp('2018-09-01')
E_COMMERCE = 'https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce'
FUNNEL = 'https://www.kaggle.com/datasets/olistbr/marketing-funnel-olist'

@lru_cache(None)
def table(name):
    filename = 'product_category_name_translation.csv' if name == 'translation' else f'olist_{name}_dataset.csv'
    d = pd.read_csv(DATA/filename)
    for c in d:
        if c.endswith(('_timestamp', '_date', '_at')):
            d[c] = pd.to_datetime(d[c], errors='coerce')
    return d

def save(name, obj):
    def convert(x):
        if isinstance(x, (np.integer, np.floating)): return x.item()
        if isinstance(x, (pd.Timestamp, Path)): return str(x)
        raise TypeError(type(x).__name__)
    (OUT/f'{name}.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert),encoding='utf8')
    print(name, 'saved')

@lru_cache(None)
def orders():
    o = table('orders').merge(table('customers')[['customer_id','customer_unique_id','customer_state']],on='customer_id',validate='many_to_one')
    amount = table('order_items').groupby('order_id').agg(value=('price','sum'),freight=('freight_value','sum'),items=('price','size'),sellers=('seller_id','nunique'))
    o = o.merge(amount,on='order_id',how='left',validate='one_to_one')
    o['delivery_days']=(o.order_delivered_customer_date-o.order_purchase_timestamp).dt.total_seconds()/86400
    o['late_days']=(o.order_delivered_customer_date.dt.normalize()-o.order_estimated_delivery_date.dt.normalize()).dt.days.clip(lower=0)
    return o

def delivered():
    o=orders()
    return o[(o.order_status=='delivered') & (o.order_purchase_timestamp >= '2017-01-01') & (o.order_purchase_timestamp < SNAPSHOT)].copy()

@lru_cache(None)
def items():
    p=table('products').merge(table('translation'),on='product_category_name',how='left',validate='many_to_one')
    d=table('order_items').merge(delivered(),on='order_id',validate='many_to_one')
    d=d.merge(p,on='product_id',how='left',validate='many_to_one')
    d['category']=d.product_category_name_english.fillna(d.product_category_name).fillna('unknown')
    return d

def customer_rfm():
    o=delivered()
    t=o.groupby('customer_unique_id').agg(last=('order_purchase_timestamp','max'),F=('order_id','nunique'),M=('value','sum'))
    t['R']=(SNAPSHOT-t['last']).dt.days
    high_spend=t.M.quantile(.75)
    t['segment']=np.select([(t.R<=90)&(t.F>=2),(t.R<=30)&(t.F==1),(t.R>90)&((t.F>=2)|(t.M>=high_spend)),t.R>90],['Повторные','Новые','Под риском','Неактивные'],default='Разовые')
    return t

CATEGORY_RU={
 'bed_bath_table':'Текстиль для дома','health_beauty':'Красота и здоровье',
 'sports_leisure':'Спорт и отдых','furniture_decor':'Мебель и декор',
 'computers_accessories':'Компьютерные аксессуары','housewares':'Товары для дома',
 'watches_gifts':'Часы и подарки','telephony':'Телефония','auto':'Автотовары',
 'toys':'Игрушки','perfumery':'Парфюмерия','baby':'Детские товары',
 'electronics':'Электроника','stationery':'Канцтовары','cool_stuff':'Разные товары',
 'garden_tools':'Садовые инструменты','pet_shop':'Зоотовары',
 'consoles_games':'Игровые приставки','luggage_accessories':'Багаж и аксессуары',
 'office_furniture':'Офисная мебель','musical_instruments':'Музыкальные инструменты',
 'fashion_bags_accessories':'Сумки и аксессуары','small_appliances':'Малая бытовая техника','market_place':'Разные товары площадки'}
def category_ru(value): return CATEGORY_RU.get(value,value.replace('_',' '))
