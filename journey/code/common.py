"""Common data preparation for the two-year, 2,500-household Kaggle release.

DAY is a relative study day, not a calendar date. SALES_VALUE is the retailer's
receipt, not necessarily the customer's payment. Manufacturer coupon discounts
are negative and are added once to SALES_VALUE to obtain customer payment.
"""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def save(name, value):
    def convert(x):
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.floating): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    (OUT / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False,
        indent=2, default=convert, allow_nan=False), encoding='utf8')

@lru_cache(None)
def table(name):
    return pd.read_csv(ROOT / 'data' / (name + '.csv'), low_memory=False)

@lru_cache(None)
def transactions():
    d = table('transaction_data').drop_duplicates().copy()
    d['paid'] = d.SALES_VALUE + d.COUPON_DISC
    d = d[(d.QUANTITY > 0) & (d.SALES_VALUE > 0) & (d.paid > 0)].copy()
    d['unit_price'] = d.paid / d.QUANTITY
    d['shelf_price'] = (d.SALES_VALUE - d.RETAIL_DISC - d.COUPON_MATCH_DISC) / d.QUANTITY
    d['discounted'] = (d.RETAIL_DISC < 0) | (d.COUPON_DISC < 0) | (d.COUPON_MATCH_DISC < 0)
    return d

@lru_cache(None)
def baskets():
    return transactions().groupby(['household_key', 'BASKET_ID'], as_index=False).agg(
        day=('DAY','min'), paid=('paid','sum'), units=('QUANTITY','sum'))

def rfm(snapshot=None, window=365):
    b = baskets()
    snapshot = int(transactions().DAY.max()) + 1 if snapshot is None else int(snapshot)
    h = b[(b.day < snapshot) & (b.day >= snapshot-window)]
    d = h.groupby('household_key').agg(R=('day', lambda s: snapshot-int(s.max())),
        F=('BASKET_ID','nunique'), M=('paid','sum'))
    for c, reverse in [('R',True),('F',False),('M',False)]:
        rank = d[c].rank(method='average', pct=True)
        d[c+'_score'] = np.clip(np.ceil((1-rank if reverse else rank)*5),1,5).astype(int)
    d['segment'] = np.select([
        (d.R_score >= 4) & (d.F_score >= 4) & (d.M_score >= 4),
        (d.R_score <= 2) & ((d.F_score >= 4) | (d.M_score >= 4)),
        d.R_score <= 2,
        (d.R_score >= 4) & (d.F_score <= 2),
    ], ['Лучшие','Под риском','Неактивные','Недавние редкие'], default='Остальные')
    return d

RU = {'FLUID MILK PRODUCTS':'Молоко', 'SOFT DRINKS':'Газированные напитки',
 'BAKED BREAD/BUNS/ROLLS':'Хлеб и булочки','BAG SNACKS':'Снеки',
 'CHEESE':'Сыр','BEEF':'Говядина','BANANAS':'Бананы','YOGURT':'Йогурт',
 'FRZN MEAT/MEAT DINNERS':'Замороженные мясные блюда', 'EGGS':'Яйца',
 'TROPICAL FRUIT':'Тропические фрукты','REFRGRATD JUICES/DRNKS':'Охлаждённые соки',
 'FRESH VEGETABLES':'Свежие овощи','COLD CEREAL':'Сухие завтраки',
 'FRZN VEGETABLE/VEG DSH':'Замороженные овощи','CHICKEN':'Курица',
 'SALAD MIX':'Салатные смеси','POTATOES':'Картофель','APPLES':'Яблоки',
 'FRESH FRUIT':'Свежие фрукты','DELI MEATS':'Мясная нарезка',
 'FRZN PIZZA':'Замороженная пицца','ICE CREAM/MILK/SHERBTS':'Мороженое',
 'COFFEE':'Кофе','PASTA SAUCE':'Соус для пасты','PASTA':'Макароны',
 'CANNED JUICES':'Консервированные соки','SUGARS/SWEETNERS':'Сахар и подсластители',
 'CARROTS':'Морковь','VEGETABLES - ALL OTHERS':'Прочие овощи'}

def category_ru(s): return RU.get(str(s), str(s))
