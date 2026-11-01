"""ЛР 10. Прогноз оттока: holdout, подбор порога только на validation."""
from common import *
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,confusion_matrix

df=pd.read_csv(DATA/'telco.csv')
df.TotalCharges=pd.to_numeric(df.TotalCharges,errors='coerce')
y=df.Churn.eq('Yes').astype(int)
x=df.drop(columns=['customerID','Churn'])
train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=42)
tr,va=train_test_split(train_idx,test_size=.25,stratify=y.iloc[train_idx],random_state=42)
num=['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']
cat=[c for c in x if c not in num]
pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore'),cat)])
model=make_pipeline(pre,LogisticRegression(max_iter=2000,random_state=42))
model.fit(x.iloc[tr],y.iloc[tr])
vp=model.predict_proba(x.iloc[va])[:,1]
thresholds=np.arange(.1,.81,.01)
threshold=float(max(thresholds,key=lambda t:f1_score(y.iloc[va],vp>=t)))
prob=model.predict_proba(x.iloc[test_idx])[:,1]
yp=(prob>=threshold).astype(int)
count=max(1,int(np.ceil(len(test_idx)*.2)))
top=np.argsort(-prob)[:count]
top_rate=float(y.iloc[test_idx].iloc[top].mean())
coef=model[-1].coef_[0];names=model[0].get_feature_names_out()
factors=sorted(zip(names,coef),key=lambda v:abs(v[1]),reverse=True)[:10]
pd.DataFrame({'customerID':df.customerID.iloc[test_idx].to_numpy(),'probability':prob,'actual':y.iloc[test_idx].to_numpy()}).to_csv(OUT/'lab10_predictions.csv',index=False)
save('lab10',dict(n=len(df),columns=len(df.columns),missing_totalcharges=int(df.TotalCharges.isna().sum()),churn_count=int(y.sum()),churn_rate=float(y.mean()),train_n=len(tr),val_n=len(va),test_n=len(test_idx),threshold=threshold,roc_auc=roc_auc_score(y.iloc[test_idx],prob),pr_auc=average_precision_score(y.iloc[test_idx],prob),precision=precision_score(y.iloc[test_idx],yp),recall=recall_score(y.iloc[test_idx],yp),f1=f1_score(y.iloc[test_idx],yp),confusion=confusion_matrix(y.iloc[test_idx],yp).tolist(),top20_rate=top_rate,test_churn_rate=float(y.iloc[test_idx].mean()),lift20=top_rate/float(y.iloc[test_idx].mean()),factors=[dict(feature=a,coef=float(b)) for a,b in factors],contract_churn=df.assign(churn=y).groupby('Contract').churn.mean().to_dict(),note='IBM Telco — учебные данные вымышленной компании. Случайный holdout, временной прогноз и причинные эффекты удержания не проверялись.'))
