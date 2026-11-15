"""Reconcile slide quantities with the original transaction rows.
Run python code/explain_results.py after lab01.py. No model is trained here.
"""
from common import *
raw=table('transaction_data');clean=transactions();b=baskets()
weekly_raw=raw.groupby('WEEK_NO').SALES_VALUE.sum().rename('raw_sales_value')
w=clean.groupby('WEEK_NO').agg(lines=('DAY','size'),sales_value=('SALES_VALUE','sum'),coupon_disc=('COUPON_DISC','sum'),paid=('paid','sum'))
w=w.join(weekly_raw);w['plotted_thousand_dollars']=w.paid/1000
saved=pd.read_csv(OUT/'lab01_weekly.csv').set_index('WEEK_NO')
assert np.allclose(w.paid,saved.paid)
assert np.allclose(w.paid,w.sales_value+w.coupon_disc)
w.to_csv(OUT/'lab01_weekly_reconciliation.csv')
example=clean[(clean.COUPON_DISC<0)&(clean.RETAIL_DISC<0)&(clean.COUPON_MATCH_DISC<0)].iloc[0]
columns=['household_key','BASKET_ID','PRODUCT_ID','DAY','WEEK_NO','QUANTITY','SALES_VALUE','RETAIL_DISC','COUPON_DISC','COUPON_MATCH_DISC','paid']
example={k:float(example[k]) if k in ['SALES_VALUE','RETAIL_DISC','COUPON_DISC','COUPON_MATCH_DISC','paid'] else int(example[k]) for k in columns}
h=clean[clean.DAY.between(622,711)].groupby('household_key').BASKET_ID.nunique()
clusters=pd.read_csv(OUT/'lab01_clusters.csv');assert len(h)==len(clusters)
assert np.array_equal(h.loc[clusters.household_key].to_numpy(),clusters.baskets.to_numpy())
sums=clean[['SALES_VALUE','COUPON_DISC','paid']].sum().to_dict()
counts=dict(raw_baskets=int(raw.BASKET_ID.nunique()),clean_baskets=len(b),households=int(clean.household_key.nunique()))
assert np.isclose(sums['paid'],8025680.07)
assert np.isclose(b.paid.median(),17.14)
save('explanations',dict(totals=sums,counts=counts,coupon_example=example,
    week20=dict(week=20,**w.loc[20].to_dict()),
    basket_definition='Rows sharing household_key and BASKET_ID form one receipt/basket. QUANTITY is units in a line, not number of distinct baskets.',
    paid_definition='paid = SALES_VALUE + COUPON_DISC after QUANTITY > 0, SALES_VALUE > 0, paid > 0.',
    median_definition='Sort one value per household or basket; use the middle value, or average of the two middle values for an even count.'))
print(json.dumps({'totals':sums,'coupon_example':example,'week20':w.loc[20].to_dict()},ensure_ascii=False,indent=2))
