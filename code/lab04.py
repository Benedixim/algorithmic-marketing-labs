"""ЛР 4. Медиамикс на ISLR Advertising. Сценарий, не причинный эффект."""
from common import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import minimize

df=pd.read_csv(DATA/'advertising.csv',index_col=0)
x=df[['TV','radio','newspaper']].copy()
x['TV_radio']=x.TV*x.radio
tr,te=train_test_split(np.arange(len(df)),test_size=.2,random_state=42)
test_model=LinearRegression().fit(x.iloc[tr],df.sales.iloc[tr])
pred=test_model.predict(x.iloc[te])
model=LinearRegression().fit(x,df.sales)
base=df[['TV','radio','newspaper']].mean().to_numpy()
budget=base.sum()
def forecast(z):
    row=pd.DataFrame([[*z,z[0]*z[1]]],columns=x.columns)
    return float(model.predict(row)[0])
opt=minimize(lambda z:-forecast(z),base,method='SLSQP',bounds=[(0,float(df[c].max())) for c in ['TV','radio','newspaper']],constraints={'type':'eq','fun':lambda z:z.sum()-budget},options={'ftol':1e-10})
assert opt.success and abs(opt.x.sum()-budget)<1e-5
save('lab04',dict(n=len(df),columns=list(df.columns),missing=int(df.isna().sum().sum()),correlations=df.corr().sales.to_dict(),test_n=len(te),test_r2=r2_score(df.sales.iloc[te],pred),test_mae=mean_absolute_error(df.sales.iloc[te],pred),budget=budget,baseline=base.tolist(),optimized=opt.x.tolist(),baseline_sales=forecast(base),optimized_sales=forecast(opt.x),change_pct=(forecast(opt.x)/forecast(base)-1)*100,coefficients=dict(zip(x.columns,model.coef_)),unit='Расходы: тысячи долларов. Продажи: тысячи единиц.',context='200 рынков, TV, радио и газеты. Конкретный товар, бренд, креативы и даты источник не раскрывает.'))
