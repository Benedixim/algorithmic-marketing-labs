"""Lab 4. Direct marketing campaigns and coupon response.

Denominator is unique addressed household/campaign pairs, numerator redeemed
at least one campaign coupon within the actual validity window. Costs are not
observed. A recommendation may allocate a conditional test budget, not claim ROI.
"""
from common import *
from math import sqrt

sent=table('campaign_table').drop_duplicates(['household_key','CAMPAIGN'])
desc=table('campaign_desc')
s=sent.merge(desc,on=['CAMPAIGN','DESCRIPTION'],validate='many_to_one')
r=table('coupon_redempt').merge(desc[['CAMPAIGN','START_DAY','END_DAY']],on='CAMPAIGN',validate='many_to_one')
r=r[r.DAY.between(r.START_DAY,r.END_DAY)]
responses=r[['household_key','CAMPAIGN']].drop_duplicates().assign(response=1)
d=s.merge(responses,on=['household_key','CAMPAIGN'],how='left',validate='one_to_one')
d['response']=d.response.fillna(0).astype(int)
groups=[]
for typ,g in d.groupby('DESCRIPTION'):
    n=len(g);k=int(g.response.sum());p=k/n;z=1.96;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    groups.append(dict(type=typ,recipients=n,responded=k,rate=p,low=center-half,high=center+half,
        campaigns=int(g.CAMPAIGN.nunique())))
campaigns=d.groupby(['CAMPAIGN','DESCRIPTION']).response.agg(['size','sum','mean']).reset_index()
campaigns.to_csv(OUT/'lab04_campaigns.csv',index=False)
save('lab04',dict(campaigns=len(desc),addressed_pairs=len(d),households=int(d.household_key.nunique()),
    raw_redemptions=len(table('coupon_redempt')),valid_redemptions=len(r),groups=groups,
    missing=s.isna().sum().astype(int).to_dict(),
    budget_rule='At equal response value, prefer A to B only if cost_A/cost_B is below response_A/response_B. Validate comparable household groups before reallocating.',
    test='Within equal prior-RFM strata randomly assign personalised vs common coupons, hold contact frequency and discount face value equal. Compare redemption, customer spend and net retailer receipt.',
    cost_thresholds={x['type']:x['rate']/next(v['rate'] for v in groups if v['type']=='TypeB') for x in groups}))
print('Lab 4',groups)
