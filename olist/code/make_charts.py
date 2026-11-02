"""Графики Matplotlib из сохранённых расчётов. Python code/run_all.py перед запуском."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter,MaxNLocator
import matplotlib.dates as mdates

FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':16,'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':15,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
BLUE='#2474AE';ORANGE='#EA8B2C'
def load(n):return json.loads((OUT/f'lab{n}.json').read_text())
def canvas():return plt.subplots(figsize=(8,4.8),layout='constrained')
def finish(fig,name):fig.savefig(FIG/f'{name}.png',dpi=170);plt.close(fig)
def bar(name,labels,values,xlabel='',percent=False):
    fig,ax=canvas();y=np.arange(len(labels));ax.barh(y,values,color=BLUE,height=.65)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.15)
    upper=max(values)*1.25;ax.set_xlim(0,upper if upper else 1)
    if percent:ax.xaxis.set_major_formatter(PercentFormatter(1))
    for pos,value in zip(y,values):
        label=f'{value*100:.1f}%' if percent else (f'{value:.2f}' if max(values)<10 else f'{value:,.0f}'.replace(',',' '))
        ax.text(value+upper*.015,pos,label,va='center',fontsize=15)
    finish(fig,name)

a=load('01')
m=pd.read_csv(OUT/'lab01_monthly.csv',index_col=0)
fig,ax=canvas();ax.bar(np.arange(len(m)),m.value/1e6,color=BLUE)
ax.set_xticks(np.arange(0,len(m),2),m.index[::2],rotation=45,ha='right');ax.set_ylabel('Стоимость товаров, млн реалов');ax.set_xlabel('Месяц покупки');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
finish(fig,'lab01_monthly')
bar('lab01_clusters',['Недавние разовые','Давние разовые','Повторные'],[a['clusters'][0]['n'],a['clusters'][1]['n'],a['clusters'][2]['n']],'Покупатели')
fig,ax=canvas();o=delivered();ax.hist(o.value.clip(upper=500),bins=np.arange(0,521,20),color=BLUE,edgecolor='white');ax.set_xlabel('Стоимость товаров в заказе, реалы');ax.set_ylabel('Заказы');ax.text(.98,.92,'Суммы от 500 объединены\nв последний столбец',ha='right',va='top',transform=ax.transAxes,fontsize=13);finish(fig,'lab01_distribution')

a=load('02');bar('lab02_hit',['Совместные покупки','Популярные категории'],[a['hit_rate'],a['baseline_rate']],'Доля клиентов с попаданием в пять рекомендаций',True)
a=load('04');channels={c['origin']:c for c in a['channels']}
keys=['organic_search','paid_search','social','email','direct_traffic']
names=['Обычный поиск','Реклама в поиске','Социальные сети','Электронные письма','Прямые переходы']
bar('lab04_leads',names,[channels[k]['leads'] for k in keys],'Заявки продавцов')
fig,ax=canvas();rates=np.array([channels[k]['rate'] for k in keys]);low=np.array([channels[k]['low'] for k in keys]);high=np.array([channels[k]['high'] for k in keys]);y=np.arange(len(keys));ax.barh(y,rates,color=BLUE);ax.errorbar(rates,y,xerr=np.array([rates-low,high-rates]),fmt='none',ecolor='black',capsize=4);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,.15);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Доля подключений за 60 дней');
for pos,v in zip(y,rates):ax.text(.145,pos,f'{v*100:.1f}%',ha='right',va='center')
finish(fig,'lab04_conversion')
a=load('05');fig,ax=canvas();ax.bar([int(k) for k in a['stars']],a['stars'].values(),color=BLUE);ax.set_xlabel('Оценка заказа в звёздах');ax.set_ylabel('Отзывы');ax.xaxis.set_major_locator(MaxNLocator(integer=True));finish(fig,'lab05_stars')
bar('lab05_topics',list(a['topics']),list(a['topics'].values()),'Негативные тексты с ключевыми словами темы')
a=load('06');p=pd.read_csv(OUT/'lab06_prices.csv',index_col=0,parse_dates=True);fig,ax=canvas();ax.plot(p.index,p.price,color=BLUE,lw=1.8);ax.axvline(pd.Timestamp(a['test_start']),color=ORANGE,linestyle=':',label='Начало проверки модели');ax.legend(fontsize=13,loc='upper right');ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'));ax.set_xlabel('Дата покупки');ax.set_ylabel('Медианная цена за день, реалы');finish(fig,'lab06_price')
fig,ax=canvas();l=a['levels'][:2];ax.bar([f"{v['price']:.2f}" for v in l],[v['units_per_selling_day'] for v in l],color=BLUE);ax.set_xlabel('Цена товара, реалы');ax.set_ylabel('Единицы на день с продажами');
for i,v in enumerate(l):ax.text(i,v['units_per_selling_day']+.05,f"{v['units_per_selling_day']:.2f}",ha='center')
ax.set_ylim(0,4.2);finish(fig,'lab06_levels')
a=load('07');bar('lab07_ratings',[r['seller_label'] for r in a['top']],[r['rating'] for r in a['top']],'Рейтинг продавца по шкале от 1 до 5')
fig,ax=canvas();rows=a['timing'];ax.bar(['В срок','С опозданием'],[v['rating'] for v in rows],color=[BLUE,ORANGE]);ax.set_ylim(0,5);ax.set_ylabel('Средняя оценка заказа');
for i,r in enumerate(rows):ax.text(i,r['rating']+.1,f"{r['rating']:.2f}",ha='center')
finish(fig,'lab07_delivery')
a=load('08');s={v['segment']:v for v in a['segments']};order=['Повторные','Новые','Разовые','Под риском','Неактивные'];bar('lab08_segments',order,[s[k]['n'] for k in order],'Покупатели')
a=load('10');r=a['recency_groups'];bar('lab10_recency',[v['group'] for v in r],[v['rate'] for v in r],'Доля продавцов без заказов следующие 60 дней',True)
bar('lab10_risk',['Все продавцы теста','Группа высокого риска'],[a['test_rate'],a['top_rate']],'Доля продавцов без заказов следующие 60 дней',True)
print('Saved',len(list(FIG.glob('*.png'))),'Matplotlib figures')
