"""Пункты 2–4. Рекомендации категорий и воспроизводимая демонстрация.
Обучение только до 1 марта 2018. Проверка на первом заказе марта–августа.
"""
from common import *
from sklearn.metrics.pairwise import cosine_similarity

d=items();cut=pd.Timestamp('2018-03-01')
tr=d[d.order_purchase_timestamp<cut];future=d[d.order_purchase_timestamp>=cut]
mat=pd.crosstab(tr.order_id,tr.category).gt(0).astype(int)
similarity=cosine_similarity(mat.T);np.fill_diagonal(similarity,0)
popular=mat.sum().to_numpy();cols=mat.columns.to_numpy();index={v:i for i,v in enumerate(cols)}
history=tr.groupby('customer_unique_id').category.agg(set)
first=future.sort_values('order_purchase_timestamp').drop_duplicates('customer_unique_id').set_index('customer_unique_id').order_id
eligible=sorted(set(history.index)&set(first.index))
actual=future[future.order_id.isin(first.loc[eligible])].groupby('customer_unique_id').category.agg(set)
def recommend(seen):
    v=np.array([int(c in seen) for c in cols]);score=similarity@v
    # Уже купленные категории оставляем. У товаров длительного использования
    # следующая покупка может быть и в той же категории.
    return cols[np.lexsort((-popular,-score))[:5]].tolist()
rows=[]
for cid in eligible:
    rec=recommend(history[cid]);base=cols[np.argsort(-popular)[:5]].tolist()
    rows.append(dict(customer=cid,hit=int(bool(set(rec)&actual[cid])),baseline=int(bool(set(base)&actual[cid])),recommendations=rec,actual=sorted(actual[cid]),history=sorted(history[cid])))
assert rows
demo=next((r for r in rows if r['hit'] and not r['baseline']),next(r for r in rows if r['hit']))
save('lab02',dict(cutoff=str(cut),train_orders=mat.shape[0],categories=mat.shape[1],eligible=len(rows),hit_rate=np.mean([r['hit'] for r in rows]),baseline_rate=np.mean([r['baseline'] for r in rows]),demo=demo,category_ru={v:category_ru(v) for v in set(demo['recommendations']+demo['history']+demo['actual'])}))
pd.DataFrame(rows).to_csv(OUT/'lab02_predictions.csv',index=False)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--customer');a=parser.parse_args()
    cid=a.customer or demo['customer']
    if cid not in history.index: raise SystemExit('Клиента нет в обучающей истории')
    print('Клиент',cid,'\nИстория',sorted(history[cid]),'\nРекомендации',recommend(history[cid]))
