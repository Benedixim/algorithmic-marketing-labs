"""Графики для презентации. Только Matplotlib, исходные расчёты в results/."""
from pathlib import Path
import json
import os
import csv
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':18,'axes.labelsize':16,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':13})
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
def read(n):return json.loads((ROOT/'results'/f'lab{n}.json').read_text())
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,facecolor='white')
    plt.close(fig)
def figure():return plt.subplots(figsize=(8,4.8),layout='constrained')
def label_bars(ax,bars,fmt='%.1f'):
    ax.bar_label(bars,fmt=fmt,padding=4,fontsize=14)

a=read('01')
fig,ax=figure()
ax.bar([x[2:] for x in a['monthly']],np.array(list(a['monthly'].values()))/1000)
ax.set(xlabel='Месяц',ylabel='Сумма покупок, тыс. £')
ax.tick_params(axis='x',rotation=45)
save(fig,'lab01_monthly')
fig,ax=figure()
counts={0:'Регулярные',1:'Недавние',2:'Давно не покупали',3:'Активные'}
rows=sorted(a['clusters'],key=lambda v:v['cluster'])
b=ax.barh([counts[v['cluster']] for v in rows],[v['n'] for v in rows])
label_bars(ax,b,'%d');ax.set(xlabel='Число клиентов',xlim=(0,1900));ax.invert_yaxis()
save(fig,'lab01_clusters')

a=read('02');fig,ax=figure()
b=ax.bar(['Популярные товары','По сходству товаров'],[a['baseline_hit5']*100,a['hit5']*100])
label_bars(ax,b,'%.1f%%');ax.set(ylabel='Клиенты с попаданием, %',ylim=(0,30))
save(fig,'lab02_hit_rate')

a=read('04')
with (ROOT/'data'/'advertising.csv').open() as f:rows=list(csv.DictReader(f))
fig,ax=figure();ax.scatter([float(r['TV']) for r in rows],[float(r['sales']) for r in rows],s=22)
ax.set(xlabel='Расходы на телевидение, тыс. $',ylabel='Продажи, тыс. единиц')
save(fig,'lab04_scatter')
fig,ax=figure();x=np.arange(3)
ax.bar(x-.19,a['baseline'],.38,label='Средние расходы')
ax.bar(x+.19,a['optimized'],.38,label='Сценарий модели')
ax.set(xticks=x,xticklabels=['Телевидение','Радио','Газеты'],ylabel='Расходы, тыс. $',ylim=(0,190));ax.legend()
save(fig,'lab04_budget')

a=read('05');fig,ax=figure()
b=ax.bar(['Позитивные','Негативные'],[a['positive'],a['negative']]);label_bars(ax,b,'%d')
ax.set(ylabel='Число уникальных фраз',ylim=(0,600))
save(fig,'lab05_classes')
fig,ax=figure();items=sorted(a['topics'].items(),key=lambda x:-x[1]);b=ax.barh([x[0] for x in items],[x[1] for x in items])
label_bars(ax,b,'%d');ax.set(xlabel='Фразы с ключевыми словами',xlim=(0,48));ax.invert_yaxis()
save(fig,'lab05_topics')

a=read('06');fig,ax=figure()
ax.plot([datetime.fromisoformat(v['date']) for v in a['sample']],[v['price'] for v in a['sample']],linewidth=1.1)
ax.set(xlabel='Дата',ylabel='Средневзвешенная цена, £',ylim=(1.4,1.74));fig.autofmt_xdate(rotation=25)
save(fig,'lab06_price')
fig,ax=figure();b=ax.bar(['Постоянная медиана','Ridge'],[a['baseline_mae'],a['mae']]);label_bars(ax,b)
ax.set(ylabel='Средняя ошибка спроса, шт.',ylim=(0,125))
save(fig,'lab06_error')

a=read('07');fig,ax=figure()
names=['Большая сумка Doily','Сумка для обеда Doily','Садовый коврик','Грелка','Большая сумка Leaf']
b=ax.barh(names,[v['score']*100 for v in a['top']]);label_bars(ax,b,'%.1f%%')
ax.set(xlabel='Нижняя граница Уилсона, %',xlim=(0,33));ax.invert_yaxis()
save(fig,'lab07_ranking')

a=read('08');items=a['summary'];fig,ax=figure()
b=ax.barh([v['segment'] for v in items],[v['customers'] for v in items]);label_bars(ax,b,'%d')
ax.set(xlabel='Число клиентов',xlim=(0,1600));ax.invert_yaxis()
save(fig,'lab08_segments')

a=read('10');fig,ax=figure()
contracts=['Month-to-month','One year','Two year']
b=ax.bar(['Помесячный','На год','На два года'],[a['contract_churn'][v]*100 for v in contracts]);label_bars(ax,b,'%.1f%%')
ax.set(ylabel='Доля клиентов с оттоком, %',ylim=(0,55))
save(fig,'lab10_contracts')
fig,ax=figure();b=ax.bar(['Весь тест','20% с высоким риском'],[a['test_churn_rate']*100,a['top20_rate']*100]);label_bars(ax,b,'%.1f%%')
ax.set(ylabel='Фактический отток, %',ylim=(0,85))
save(fig,'lab10_risk')
print('Построено 13 графиков Matplotlib в figures/.')
