"""Lab 10. Prospective customer inactivity prediction with time-split evaluation.

Each row uses only the 180 days before a snapshot. Target is no recorded purchase
in the next 60 days, not confirmed contract termination. Snapshots have disjoint
target windows; threshold is selected on validation only.
"""
from common import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score

b=baskets();cohort=b[b.day<180].household_key.unique()
def snapshot(day):
    h=b[(b.day<day)&(b.day>=day-180)]
    d=h.groupby('household_key').agg(R=('day',lambda x:day-int(x.max())),
        F=('BASKET_ID','nunique'),M=('paid','sum'),basket_value=('paid','mean'))
    d=d.reindex(cohort).fillna({'R':181,'F':0,'M':0,'basket_value':0})
    recent=b[(b.day<day)&(b.day>=day-60)].groupby('household_key').BASKET_ID.nunique()
    previous=b[(b.day<day-60)&(b.day>=day-120)].groupby('household_key').BASKET_ID.nunique()
    d['recent_trips']=recent.reindex(cohort).fillna(0);d['change']=d.recent_trips-previous.reindex(cohort).fillna(0)
    returning=set(b[(b.day>=day)&(b.day<day+60)].household_key)
    d['inactive']=[int(k not in returning) for k in d.index];d['snapshot']=day
    return d
train=pd.concat([snapshot(s) for s in [300,360,420]])
valid=snapshot(500);test=snapshot(620)
features=['R','F','M','basket_value','recent_trips','change']
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42))
model.fit(train[features],train.inactive)
pv=model.predict_proba(valid[features])[:,1]
threshold=max(np.arange(.05,.96,.01),key=lambda v:f1_score(valid.inactive,pv>=v,zero_division=0))
pt=model.predict_proba(test[features])[:,1];pred=pt>=threshold
topn=int(np.ceil(len(test)*.2));top=np.argsort(-pt)[:topn]
groups=[]
for label,mask in [('До 7 дней',test.R<=7),('От 8 до 30 дней',test.R.between(8,30)),('Более 30 дней',test.R>30)]:
    z=test[mask];groups.append(dict(group=label,n=len(z),rate=float(z.inactive.mean())))
test['probability']=pt;test['selected']=pred
test.to_csv(OUT/'lab10_holdout.csv')
save('lab10',dict(train=len(train),valid=len(valid),test=len(test),cohort=len(cohort),train_snapshots=[300,360,420],
    validation_snapshot=500,test_snapshot=620,horizon=60,threshold=float(threshold),
    test_inactive=int(test.inactive.sum()),test_rate=float(test.inactive.mean()),
    auc=float(roc_auc_score(test.inactive,pt)),average_precision=float(average_precision_score(test.inactive,pt)),
    precision=float(precision_score(test.inactive,pred,zero_division=0)),recall=float(recall_score(test.inactive,pred,zero_division=0)),
    f1=float(f1_score(test.inactive,pred,zero_division=0)),top_n=topn,top_rate=float(test.iloc[top].inactive.mean()),
    recency_groups=groups,coefficients=dict(zip(features,map(float,model[1].coef_[0]))),
    missing_after_aggregation=int(test[features].isna().sum().sum()),
    retention='For high-risk previously frequent shoppers send a reminder of usual categories. Randomise eligible households to reminder versus no reminder and measure 60-day return and spend. No invented discount or measured causal uplift.'))
print('Lab 10 AUC',roc_auc_score(test.inactive,pt),'top20%',test.iloc[top].inactive.mean())
