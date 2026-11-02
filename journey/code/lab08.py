"""Lab 8. RFM segmentation with reproducible quantile rules."""
from common import *
d=rfm();t=transactions();snapshot=int(t.DAY.max())+1
groups=[]
for name,g in d.groupby('segment'):
    groups.append(dict(segment=name,n=len(g),R=float(g.R.median()),F=float(g.F.median()),M=float(g.M.median())))
d.to_csv(OUT/'lab08_rfm.csv')
save('lab08',dict(snapshot=snapshot,window=365,households=len(d),groups=groups,
    rule='Each component is scored 1 to 5 by average percentile rank, reverse R. Best R,F,M>=4; risk R<=2 and F>=4 or M>=4; inactive R<=2; recent rare R>=4 and F<=2; others remaining. Priority is as listed.',
    thresholds={c:d[c].quantile([.2,.4,.6,.8]).to_dict() for c in ['R','F','M']}))
print('Lab 8',groups)
