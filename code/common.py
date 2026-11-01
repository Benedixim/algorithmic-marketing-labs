"""Общие функции. Python 3.10+, запускать скрипты из любой папки."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)

def save(name, result):
    (OUT / f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x,'item') else str(x)), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

def retail(raw=False):
    cache = DATA / 'retail.pkl'
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = pd.read_excel(DATA / 'Online Retail.xlsx', dtype={'InvoiceNo':str,'StockCode':str})
        df.to_pickle(cache)
    if raw:
        return df
    # Только реальные товарные покупки с известным клиентом.
    return df.loc[df.CustomerID.notna() & (df.Quantity>0) & (df.UnitPrice>0)
                  & ~df.InvoiceNo.str.startswith('C')
                  & df.StockCode.str.match(r'^\d{5}[A-Za-z]?$')].drop_duplicates().copy()

def rfm(df, date='2011-12-10'):
    df = df.copy()
    df['revenue'] = df.Quantity * df.UnitPrice
    t = df.groupby('CustomerID').agg(last=('InvoiceDate','max'), F=('InvoiceNo','nunique'), M=('revenue','sum'))
    t['R'] = (pd.Timestamp(date)-t['last']).dt.days
    return t[['R','F','M']]

def segments(df):
    t = rfm(df)
    # rank(method='average') сохраняет одинаковые баллы для одинаковых значений.
    t['Rs'] = (6-np.ceil(t.R.rank(pct=True)*5)).clip(1,5).astype(int)
    t['Fs'] = np.ceil(t.F.rank(pct=True)*5).clip(1,5).astype(int)
    t['Ms'] = np.ceil(t.M.rank(pct=True)*5).clip(1,5).astype(int)
    first = df.groupby('CustomerID').InvoiceDate.min()
    recently_first = (pd.Timestamp('2011-12-10')-first).dt.days<=60
    t['segment'] = np.select([
        (t.Rs>=4)&(t.Fs>=4)&(t.Ms>=4),
        (t.Rs<=2)&(t.Fs>=3),
        (t.Rs>=4)&(t.F<=2)&recently_first,
        t.Rs<=2,
    ], ['Лояльные','Под риском','Новые','Спящие'], default='Остальные')
    return t
