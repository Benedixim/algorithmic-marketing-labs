"""All slide charts use Matplotlib and the saved real analysis outputs."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':16,'axes.labelsize':17,
    'xtick.labelsize':14,'ytick.labelsize':15,'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
    'savefig.facecolor':'white'})
BLUE='#2474AE';ORANGE='#EA8B2C'
def load(n):return json.loads((OUT/('lab'+n+'.json')).read_text())
def canvas():return plt.subplots(figsize=(8,4.8),layout='constrained')
def finish(fig,name):fig.savefig(FIG/(name+'.png'),dpi=170);plt.close(fig)
def bar(name,labels,values,xlabel,percent=False,precision=None):
    fig,ax=canvas();y=np.arange(len(labels));ax.barh(y,values,color=BLUE,height=.65)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel(xlabel)
    ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
    upper=max(values)*1.25 if max(values)>0 else 1;ax.set_xlim(0,upper)
    if percent:ax.xaxis.set_major_formatter(PercentFormatter(1))
    for pos,v in zip(y,values):
        label=f'{v*100:.1f}%' if percent else f'{v:.{precision}f}' if precision is not None else f'{v:,.0f}'.replace(',',' ')
        ax.text(v+upper*.015,pos,label,va='center',fontsize=15)
    finish(fig,name)

a=load('01');w=pd.read_csv(OUT/'lab01_weekly.csv')
fig,ax=canvas();ax.plot(w.WEEK_NO,w.paid/1000,color=BLUE,lw=2)
ax.set_xlabel('Неделя исследования');ax.set_ylabel('Покупки семей выборки, тыс. долларов');ax.grid(alpha=.15)
finish(fig,'lab01_weekly')
bar('lab01_clusters',[v['name'] for v in a['clusters']],[v['n'] for v in a['clusters']],'Домохозяйства с покупками за 90 дней')
a=load('02');bar('lab02_quality',['По истории и связям товаров','Популярные товары'],[a['hit_rate'],a['baseline_rate']],'Попадание в пять рекомендаций',True)
a=load('04');names=['Персональные\nкупоны A','Общие\nкупоны B','Общие\nкупоны C']
bar('lab04_response',names,[v['rate'] for v in a['groups']],'Семьи с погашением купона среди получателей',True)
bar('lab04_cost',names,[a['cost_thresholds'][v['type']] for v in a['groups']],'Предел стоимости контакта относительно кампании B',precision=2)
a=load('05');matrix=np.array(a['confusion'])
fig,ax=canvas();ax.imshow(matrix,cmap='Blues',vmin=0,vmax=210)
ax.set_xticks([0,1],['Негативный','Положительный']);ax.set_yticks([0,1],['Негативный','Положительный'])
ax.set_xlabel('Ответ модели');ax.set_ylabel('Отзыв по оценке автора')
for y in range(2):
    for x in range(2):ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=28,fontweight='bold',color='white' if matrix[y,x]>120 else 'black')
ax.spines[['top','right','bottom','left']].set_visible(False);ax.tick_params(length=0)
finish(fig,'lab05_confusion')
bar('lab05_topics',[v['theme'].replace('Поломки и отказ в работе','Поломки и отказ\nв работе') for v in a['topics']],[v['count'] for v in a['topics']],'Негативные отзывы с упоминанием темы')
a=load('06');w=pd.read_csv(OUT/'lab06_weekly.csv')
fig,ax=canvas();ax.plot(w.week_number,w.price,color=BLUE,lw=2)
ax.axvline(a['test_start'],color=ORANGE,linestyle=':',label='Начало проверки прогноза');ax.legend(fontsize=13)
ax.set_xlabel('Неделя исследования');ax.set_ylabel('Цена упаковки молока, доллары');ax.grid(alpha=.15)
finish(fig,'lab06_price')
bar('lab06_quantity',[f"{v['price']:.2f} доллара" for v in a['bands']],[v['units'] for v in a['bands']],'Упаковки в неделю среди семей выборки',precision=2)
a=load('07');bar('lab07_ranking',[v['category_ru']+'\n'+str(v['PRODUCT_ID']) for v in a['top']],[v['penetration'] for v in a['top']],'Доля семей, купивших конкретный товар',True)
bar('lab07_factors',['Прежний охват покупателей','Число магазинов','Доля покупок со скидкой','Цена'],[a['importances'][k] for k in ['prior_households','stores','discount_share','price']],'Важность признака для прогноза популярности',True)
a=load('08');order=['Лучшие','Под риском','Неактивные','Недавние редкие','Остальные'];g={v['segment']:v for v in a['groups']}
bar('lab08_segments',order,[g[k]['n'] for k in order],'Домохозяйства с покупками за последний год')
a=load('10');bar('lab10_recency',[v['group'] for v in a['recency_groups']],[v['rate'] for v in a['recency_groups']],'Доля без покупок в следующие 60 дней',True)
bar('lab10_risk',['Все семьи проверки','20% с наибольшим риском'],[a['test_rate'],a['top_rate']],'Доля без покупок в следующие 60 дней',True)
print('Created',len(list(FIG.glob('*.png'))),'Matplotlib figures')
