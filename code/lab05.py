"""ЛР 5. Тональность отзывов Amazon Cells: TF-IDF + LogisticRegression."""
from common import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

rows=[line.rsplit('\t',1) for line in (DATA/'amazon_cells_labelled.txt').read_text().splitlines() if line]
raw=pd.DataFrame(rows,columns=['text','label'])
raw.label=raw.label.astype(int)
df=raw.drop_duplicates('text').copy()
tr,te=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2),LogisticRegression(max_iter=1000,random_state=42))
pipe.fit(tr.text,tr.label)
pred=pipe.predict(te.text)
negative=df[df.label==0]
topics={'Звук':r'\b(sound|hear|volume|audio|voice)\b','Качество':r'\b(broke|broken|quality|poor|plastic|defective)\b','Батарея':r'\b(battery|charge|charging)\b','Удобство':r'\b(fit|ear|ears|comfortable|uncomfortable)\b'}
counts={k:int(negative.text.str.contains(v,case=False,regex=True).sum()) for k,v in topics.items()}
vocab=pipe[0].get_feature_names_out(); coef=pipe[1].coef_[0]
save('lab05',dict(raw_n=len(raw),n=len(df),removed_duplicates=len(raw)-len(df),train_n=len(tr),test_n=len(te),fields=['text: str','label: int (0/1)'],missing=int(df.isna().sum().sum()),positive=int(df.label.sum()),negative=int((df.label==0).sum()),accuracy=accuracy_score(te.label,pred),f1=f1_score(te.label,pred),confusion=confusion_matrix(te.label,pred).tolist(),topics=counts,negative_terms=vocab[np.argsort(coef)[:8]].tolist(),positive_terms=vocab[np.argsort(-coef)[:8]].tolist(),note='В исходном наборе классы искусственно сбалансированы. Доля негатива не характеризует весь Amazon. Темы могут пересекаться.'))
