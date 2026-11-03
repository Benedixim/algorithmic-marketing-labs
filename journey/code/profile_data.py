"""Descriptive data profiles used before the laboratory results.

Statistics describe the original two-year Kaggle transaction table and the
original 2,000 positive/negative electronics reviews. Identifier medians are not
reported; their plots show the distribution of record counts per identifier.
Numeric histograms collect values above the 99th percentile in the last bin;
full minima and maxima are retained in the adjacent statistics and JSON.
"""
from common import ROOT, OUT, table, save
from sentiment_common import reviews
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':12,
                     'figure.facecolor':'white','axes.facecolor':'white',
                     'savefig.facecolor':'white'})

def number(v, decimals=0):
    if decimals:
        return f'{v:,.{decimals}f}'.replace(',',' ').replace('.',',').replace('-', '−')
    return f'{v:,.0f}'.replace(',',' ').replace('-', '−')

def summary(values, decimals=0):
    x=pd.Series(values).dropna().astype(float)
    return 'Медиана '+number(x.median(),decimals)+'\nОт '+number(x.min(),decimals)+' до '+number(x.max(),decimals)

def hist(values, name, discrete=False, label='',wide=False):
    x=pd.Series(values).dropna().astype(float)
    fig,ax=plt.subplots(figsize=((7.2 if wide else 3.8),.85))
    fig.subplots_adjust(left=(.11 if wide else .20),right=(.93 if wide else .85),bottom=.44,top=.91)
    if discrete:
        counts=x.value_counts().sort_index()
        heights=counts.values/len(x)*100
        ax.bar(counts.index,heights,width=.7,color='#2474AE')
        ax.set_xticks(range(int(x.min()),int(x.max())+1))
    else:
        cap=float(x.quantile(.99))
        shown=x.clip(upper=cap)
        bins=min(16,max(1,int(shown.nunique())))
        heights,_,_=ax.hist(shown,bins=bins,weights=np.full(len(shown),100/len(shown)),color='#2474AE',edgecolor='white',linewidth=.45)
        lo=float(shown.min());hi=float(shown.max())
        if lo<hi:
            ticks=[lo,(lo+hi)/2,hi]
            def tick(v):
                if abs(v-round(v))<.02:return number(round(v))
                return f'{v:.1f}'.replace('.',',').replace('-','−')
            labels=[tick(v) for v in ticks]
            if cap<float(x.max()):labels[-1]='≥'+labels[-1]
            ax.set_xticks(ticks,labels)
    ymax=max(10,float(np.ceil(max(heights)/10)*10))
    ax.set_ylim(0,ymax)
    ax.set_yticks([0,ymax],['0%',number(ymax)+'%'])
    ax.tick_params(axis='both',labelsize=14,length=2,pad=1)
    ax.set_xlabel(label,fontsize=13,labelpad=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_linewidth(.6)
    fig.savefig(FIG/(name+'.png'),dpi=200)
    plt.close(fig)

def entry(data,key,description,kind,plot_values=None,plot_label='',decimals=0,discrete=False):
    s=data[key]
    missing=int(s.isna().sum())
    if s.dtype=='object': missing+=int(s.fillna('x').str.strip().eq('').sum())
    name='profile_journey_'+key.lower()
    values=s if plot_values is None else plot_values
    wide=key in ['QUANTITY','SALES_VALUE','RETAIL_DISC','COUPON_DISC','COUPON_MATCH_DISC','TRANS_TIME']
    axis_label=plot_label or ('День' if key=='DAY' else 'Неделя' if key=='WEEK_NO' else 'Количество' if key=='QUANTITY' else 'Доллары')
    hist(values,name,discrete,axis_label,wide)
    valid=s.dropna()
    result=dict(key=key,description=description,type=kind,missing=missing,
                unique=int(valid.nunique()),figure=name,plot_label=plot_label)
    if plot_values is not None:
        result['stats']=summary(values)
        result['record_count_median']=float(pd.Series(values).median())
    else:
        result.update(min=float(valid.min()),median=float(valid.median()),max=float(valid.max()))
        if key=='TRANS_TIME':
            v=int(valid.median());median=f'{v//100:02d}.{v%100:02d}'
            result['stats']='Медиана '+median+'\n'+number(valid.nunique())+' значений'
        else:
            result['stats']='Медиана '+number(valid.median(),decimals)+'\nОт '+number(valid.min(),decimals)+' до '+number(valid.max(),decimals)
    return result

raw=table('transaction_data')
id_fields=[('household_key','Код семьи','Строк на семью'),
           ('BASKET_ID','Код корзины','Позиций в корзине'),
           ('PRODUCT_ID','Код товара','Строк на товар'),
           ('STORE_ID','Код магазина','Строк на магазин')]
rows1=[entry(raw,key,desc,'Целое',raw.groupby(key).size(),label) for key,desc,label in id_fields]
rows1.extend([entry(raw,'DAY','День исследования','Целое'),
              entry(raw,'WEEK_NO','Неделя исследования','Целое')])
rows2=[entry(raw,'QUANTITY','Количество товара','Целое'),
       entry(raw,'SALES_VALUE','Поступление магазину','Дробное',decimals=2),
       entry(raw,'RETAIL_DISC','Скидка магазина','Дробное',decimals=2),
       entry(raw,'COUPON_DISC','Купон производителя','Дробное',decimals=2),
       entry(raw,'COUPON_MATCH_DISC','Доплата по купону','Дробное',decimals=2),
       entry(raw,'TRANS_TIME','Часы и минуты','Целое',raw.TRANS_TIME//100,'Час покупки')]
# Time is numeric HHMM, but neither its cardinality nor its median is an ID statistic.
time=rows2[-1];v=int(raw.TRANS_TIME.median())
clock=lambda n:f'{int(n)//100:02d}.{int(n)%100:02d}'
time['stats']='Медиана '+clock(v)+'\nОт '+clock(raw.TRANS_TIME.min())+' до '+clock(raw.TRANS_TIME.max())
time['median']=float(raw.TRANS_TIME.median())

current=raw.drop_duplicates().copy()
flow=[dict(label='Исходные строки',n=len(raw),removed=0)]
for label,mask in [
    ('Количество больше нуля',lambda d:d.QUANTITY>0),
    ('Поступление больше нуля',lambda d:d.SALES_VALUE>0),
    ('Оплата больше нуля',lambda d:(d.SALES_VALUE+d.COUPON_DISC)>0)]:
    before=len(current);current=current[mask(current)].copy()
    flow.append(dict(label=label,n=len(current),removed=before-len(current)))
current['paid']=current.SALES_VALUE+current.COUPON_DISC
expected=pd.read_json(OUT/'lab01.json',typ='series')
assert len(current)==int(expected.clean_rows)
households=current.groupby('household_key').agg(baskets=('BASKET_ID','nunique'),paid=('paid','sum'),products=('PRODUCT_ID','nunique'))
correlations=households.corr().to_dict()
journey=dict(raw_rows=len(raw),columns=len(raw.columns),fields=rows1+rows2,
    page1=rows1,page2=rows2,duplicates=int(raw.duplicated().sum()),
    flow=flow,retained_share=len(current)/len(raw),
    correlations_by_household=correlations,
    transaction_correlation=raw.select_dtypes(include='number').corr().to_dict(),
    all_tables=[dict(name=name,rows=len(table(name)),columns=len(table(name).columns))
                for name in ['transaction_data','product','hh_demographic','campaign_desc','campaign_table','coupon','coupon_redempt']])

all_reviews=reviews()
amazon=all_reviews[(all_reviews.domain=='electronics') & all_reviews.subset.isin(['positive','negative'])].copy()
amazon['words']=amazon.review_text.str.findall(r'\b\w+\b').str.len()
amazon['title_words']=amazon.title.str.findall(r'\b\w+\b').str.len()
amazon['name_length']=amazon.product_name.str.len()
arows=[]
spec=[('asin','Код товара Amazon','Текст',amazon.groupby('asin').size(),'Отзывы на товар'),
      ('product_name','Название товара','Текст',amazon.name_length,'Символы в названии'),
      ('rating','Оценка автора','Число',amazon.rating,'Звёзды'),
      ('date','Дата публикации','Дата',amazon.date.dropna().dt.year,'Год публикации'),
      ('title','Заголовок отзыва','Текст',amazon.title_words,'Слова в заголовке'),
      ('review_text','Текст отзыва','Текст',amazon.words,'Слова в отзыве'),
      ('helpful','Голоса о полезности','Текст',None,'Голоса за отзыв')]
for key,desc,kind,values,label in spec:
    s=amazon[key];missing=int(s.isna().sum())
    if s.dtype=='object':missing+=int(s.fillna('x').str.strip().eq('').sum())
    fig_name='profile_amazon_'+key
    row=dict(key=key,description=desc,type=kind,missing=missing,unique=int(s.nunique()),figure=fig_name,plot_label=label)
    if key=='helpful':
        values=pd.to_numeric(s.str.extract(r'^(\d+)')[0],errors='coerce')
    row['stats']=summary(values,1 if key=='rating' else 0)
    if key=='date':row['stats']='Медиана '+str(int(pd.Series(values).median()))+'\nОт '+str(int(pd.Series(values).min()))+' до '+str(int(pd.Series(values).max()))
    row['distribution_median']=float(pd.Series(values).median())
    hist(values,fig_name,key=='rating',label)
    arows.append(row)
no_empty=amazon[amazon.norm.ne('') & amazon.asin.ne('') & amazon.rating.isin([1,2,4,5])]
unique=no_empty.drop_duplicates('norm')
assert len(unique)==1965
import json
lab05=json.loads((OUT/'lab05.json').read_text())
amazon_profile=dict(archive_rows=len(all_reviews),selected_rows=len(amazon),used_columns=7,original_fields=11,
    fields=arows,date_missing=int(amazon.date.isna().sum()),
    stars={str(int(k)):int(v) for k,v in amazon.rating.value_counts().sort_index().items()},
    word_rating_correlation=float(amazon[['words','rating']].corr().iloc[0,1]),
    flow=[dict(label='Отзывы об электронике',n=len(amazon),removed=0),
          dict(label='Заполнены текст и оценка',n=len(no_empty),removed=len(amazon)-len(no_empty)),
          dict(label='Без повторов текста',n=len(unique),removed=len(no_empty)-len(unique))],
    train=lab05['train'],test=lab05['test'],date_min=str(amazon.date.min().date()),date_max=str(amazon.date.max().date()))
save('data_profiles',dict(journey=journey,amazon=amazon_profile))
print('Profiles created, raw Kaggle rows',len(raw),'clean',len(current),
      'Amazon selected',len(amazon),'unique',len(unique))
print('Cleaning counts',flow)
print('Household baskets/spend Pearson',households.baskets.corr(households.paid))
