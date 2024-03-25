import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime 
from sklearn.cluster import KMeans ,DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import json

NOW = datetime.now()
with open("data/sample.json","r",encoding = "utf-8") as f: 
    data = json.load(f)

data = [{'pk':a['pk'],'customer_ID':str(int(a['fields']['phone'])),'price':int(a['fields']['price']),'date':datetime.strptime(a['fields']['date'][:10],"%Y-%m-%d")} for a in data if a["fields"]["order_condition"] == "4"]

for index , d in enumerate(data):
    if d["customer_ID"][0:2] == "98":
        data[index]["customer_ID"] = d["customer_ID"][2:]
    if not len(d["customer_ID"]) == 10:
        data.pop(index)
            
df = pd.DataFrame(data)
print(df.head())
# df.to_csv('Factor_edited.csv', index=False)

# Recency Table
df_recency = df.groupby(['customer_ID'],as_index=False)['date'].max()
df_recency.columns = ['customer_ID','Last_Purchase_Date']
df_recency['Recency'] = df_recency.Last_Purchase_Date.apply(lambda x:(NOW - x).days)
df_recency.drop(columns=['Last_Purchase_Date'],inplace=True)

print(df_recency)
# FM Table
FM_Table = df.groupby('customer_ID').agg({'pk': lambda x:len(x),'price': lambda x:x.sum()})
FM_Table.rename(columns = {'pk' :'Frequency','price':'Monetary'},inplace= True) 

# RFM Table
RFM_Table = df_recency.merge(FM_Table,left_on='customer_ID',right_on='customer_ID')
print(RFM_Table)

# Split into metrics and Scores

RFM_Table = RFM_Table.copy()

r_labels = range(4, 0, -1)
r_groups = pd.qcut(RFM_Table['Recency'], q=4, labels=r_labels)
f_labels = range(1,5)
normalized = preprocessing.normalize([RFM_Table['Frequency']])
f_groups = pd.cut(normalized[0], bins=4, labels=f_labels)
m_labels = range(1,5)
m_groups = pd.qcut(RFM_Table['Monetary'], q=4, labels=m_labels)
RFM_Table['R'] = r_groups
RFM_Table['F'] = f_groups
RFM_Table['M'] = m_groups
print(RFM_Table)


X = RFM_Table[['R','F','M']]
kmean= KMeans(n_clusters=3)
kmean.fit(X)
RFM_Table['KCluster'] = kmean.labels_

print(RFM_Table)

sns.barplot(data=RFM_Table, x='KCluster', y='Recency')

sns.barplot(data=RFM_Table, x='KCluster', y='Monetary')

sns.barplot(data=RFM_Table, x='KCluster', y='Frequency')

# Find Best K with Inertia

wcss = {}
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(X)
    wcss[k] = km.inertia_
    
sns.pointplot(x=list(wcss.keys()), y=list(wcss.values()))

# Evaluation Model

silhouette_score(X , kmean.labels_)

# Simple DBscan model

DBS=DBSCAN(eps=0.7, min_samples=3)
DBS.fit(X)

np.unique(DBS.labels_)

# Evaluation Model

silhouette_score(X, DBS.labels_)

