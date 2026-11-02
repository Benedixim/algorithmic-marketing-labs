"""Lab 2. Item recommendations; chronological holdout and a runnable demo.

python code/lab02.py --household 1
Repeat purchases are valid recommendations for grocery shoppers.
"""
from common import *
from scipy.sparse import csr_matrix
import argparse

parser=argparse.ArgumentParser();parser.add_argument('--household',type=int,default=1)
args=parser.parse_args()
t=transactions(); cutoff=650
train=t[t.DAY<=cutoff].copy(); future=t[t.DAY>cutoff]
popular=train.groupby('PRODUCT_ID').BASKET_ID.nunique().sort_values(ascending=False)
products=popular.head(2000).index.to_numpy(); pos={p:i for i,p in enumerate(products)}
tr=train[train.PRODUCT_ID.isin(products)].copy()
bids=pd.Index(tr.BASKET_ID.unique()); rows=bids.get_indexer(tr.BASKET_ID)
matrix=csr_matrix((np.ones(len(tr),dtype=np.float32),(rows,tr.PRODUCT_ID.map(pos))),shape=(len(bids),len(products)))
matrix.data[:]=1
co=(matrix.T@matrix).toarray();norm=np.sqrt(np.outer(np.diag(co),np.diag(co)))
similarity=np.divide(co,norm,out=np.zeros_like(co),where=norm>0);np.fill_diagonal(similarity,0)
households=pd.Index(sorted(train.household_key.unique()))
profile=csr_matrix((np.ones(len(tr),dtype=np.float32),(households.get_indexer(tr.household_key),tr.PRODUCT_ID.map(pos))),shape=(len(households),len(products)))
weights=profile.toarray(); weights=np.log1p(weights)
scores=weights@similarity
metadata=table('product').set_index('PRODUCT_ID')
def recommend(household):
    if household not in households: values=np.arange(5)
    else: values=np.argsort(-scores[households.get_loc(household)],kind='stable')[:5]
    return [dict(product=int(products[i]),category=metadata.loc[products[i],'COMMODITY_DESC'],
                 category_ru=category_ru(metadata.loc[products[i],'COMMODITY_DESC'])) for i in values]

last=future.sort_values(['DAY','TRANS_TIME','BASKET_ID']).groupby('household_key').BASKET_ID.last()
eligible=last[last.index.isin(households)]
truth=future[future.BASKET_ID.isin(eligible)].groupby('household_key').PRODUCT_ID.agg(set)
hits=[];baseline=[]
for h,actual in truth.items():
    rec={r['product'] for r in recommend(h)}
    hits.append(bool(rec&actual));baseline.append(bool(set(products[:5])&actual))
demos=[]
for h in list(dict.fromkeys([args.household,1,100,1000])):
    history=train[train.household_key==h].groupby('PRODUCT_ID').BASKET_ID.nunique().nlargest(3)
    demos.append(dict(household=int(h),history=[dict(product=int(p),category_ru=category_ru(metadata.loc[p,'COMMODITY_DESC'])) for p in history.index],recommendations=recommend(h)))
save('lab02',dict(cutoff=cutoff,train_rows=len(train),candidate_products=len(products),train_baskets=len(bids),
    test_households=len(truth),hit_rate=float(np.mean(hits)),baseline_rate=float(np.mean(baseline)),
    definition='At least one held-out last-basket product among five recommendations. All actual products retained in truth.',
    train_end=int(train.DAY.max()),test_start=int(future.DAY.min()),demos=demos))
print(json.dumps(demos[0],ensure_ascii=False,indent=2))
print('Lab 2 holdout',len(truth),'hit',np.mean(hits),'popularity',np.mean(baseline))
