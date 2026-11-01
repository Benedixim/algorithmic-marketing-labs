"""ЛР 2. Item-to-item рекомендации и временная проверка против популярности."""
from common import *
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

df=retail()
cut=pd.Timestamp('2011-10-01')
train=df[df.InvoiceDate<cut]
test=df[df.InvoiceDate>=cut]
products=sorted(train.StockCode.unique())
idx={v:i for i,v in enumerate(products)}
invoices={v:i for i,v in enumerate(train.InvoiceNo.unique())}
pairs=train[['InvoiceNo','StockCode']].drop_duplicates()
b=csr_matrix((np.ones(len(pairs)),(pairs.InvoiceNo.map(invoices),pairs.StockCode.map(idx))),shape=(len(invoices),len(idx)))
v=normalize(b,axis=0)
sim=(v.T@v).tocsr()
sim.setdiag(0)
pop=np.asarray(b.sum(axis=0)).ravel()
hist=train.groupby('CustomerID').StockCode.apply(set).to_dict()
# Первая будущая корзина каждого известного клиента.
first=test.sort_values('InvoiceDate').groupby('CustomerID').InvoiceNo.first()
future=test[test.InvoiceNo.isin(first)].groupby('CustomerID').StockCode.apply(set)
records=[]
examples=[]
names=df.drop_duplicates('StockCode').set_index('StockCode').Description.to_dict()
for cid,truth in future.items():
    if cid not in hist: continue
    seen=[idx[p] for p in hist[cid] if p in idx]
    target={idx[p] for p in truth-hist[cid] if p in idx}
    if not target: continue
    score=np.asarray(sim[seen].sum(axis=0)).ravel()
    base=pop.copy()
    score[seen]=-np.inf
    base[seen]=-np.inf
    rec=np.argsort(-score,kind='stable')[:5]
    ref=np.argsort(-base,kind='stable')[:5]
    hit=len(set(rec)&target); hb=len(set(ref)&target)
    records.append(dict(customer=int(cid),recall=hit/len(target),baseline_recall=hb/len(target),hit=int(hit>0),baseline_hit=int(hb>0),targets=len(target)))
    examples.append(dict(customer=int(cid),hit=hit,history=[names[products[i]] for i in sorted(seen)[:5]],recommendations=[dict(code=products[i],name=names[products[i]]) for i in rec],actual=[names[products[i]] for i in sorted(target)]))
r=pd.DataFrame(records)
r.to_csv(OUT/'lab02_test.csv',index=False)
demo=next((e for e in examples if e['hit']>0),examples[0])
save('lab02',dict(cutoff=str(cut.date()),train_rows=len(train),test_rows=len(test),evaluated=len(r),
    recall5=r.recall.mean(),baseline_recall5=r.baseline_recall.mean(),hit5=r.hit.mean(),baseline_hit5=r.baseline_hit.mean(),demo=demo,
    note='Оценка только на известных клиентах с новыми доступными товарами в первой будущей корзине. Повторные покупки исключены.'))
