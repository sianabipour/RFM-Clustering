import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime 
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from defines import RScore, FMScore
# from feature_engine.outlier_removers import Winsorizer
import json
NOW = datetime.now()
with open("Factor.json","r",encoding = "utf-8") as f: 
    data = json.load(f)

data = [{'pk':a['pk'],'customer_ID':str(int(a['fields']['phone'])),'price':a['fields']['price'],'date':datetime.strptime(a['fields']['date'][:10],"%Y-%m-%d")} for a in data if a["fields"]["order_condition"] == "4"]

for index , d in enumerate(data):
    if d["customer_ID"][0:2] == "98":
        data[index]["customer_ID"] = d["customer_ID"][2:]
    if not len(d["customer_ID"]) == 10:
        data.pop(index)
            
df = pd.DataFrame(data)
# print(df.describe(include='O'))
# df.to_csv('Factor_edited.csv', index=False)

# Recency Table
df_recency = df.groupby(['customer_ID'],as_index=False)['date'].max()
df_recency.columns = ['customer_ID','Last_Purchase_Date']
df_recency['Recency'] = df_recency.Last_Purchase_Date.apply(lambda x:(NOW - x).days)
df_recency.drop(columns=['Last_Purchase_Date'],inplace=True)

# FM Table
FM_Table = df.groupby('customer_ID').agg({'pk': lambda x:len(x),'price': lambda x:x.sum()})
FM_Table.rename(columns = {'pk' :'Frequency','price':'Monetary_Value'},inplace= True) 

# RFM Table
RFM_Table = df_recency.merge(FM_Table,left_on='customer_ID',right_on='customer_ID')


# Split into metrics and Scores

quantiles = RFM_Table.quantile(q=[0.25,0.50,0.75])
quantiles = quantiles.to_dict()

segmented_rfm = RFM_Table.copy()

segmented_rfm['R_quartile'] = segmented_rfm['Recency'].apply(RScore, args=('Recency',quantiles))
segmented_rfm['F_quartile'] = segmented_rfm['Frequency'].apply(FMScore, args=('Frequency',quantiles))
segmented_rfm['M_quartile'] = segmented_rfm['Monetary_Value'].apply(FMScore, args=('Monetary_Value',quantiles))
segmented_rfm['RFM_Segment'] = segmented_rfm.R_quartile.map(str)+segmented_rfm.F_quartile.map(str)+segmented_rfm.M_quartile.map(str)
segmented_rfm['RFM_Score'] = segmented_rfm[['R_quartile','F_quartile','M_quartile']].sum(axis=1)

print(segmented_rfm.head())

# Customer Analysis

# Suggested marketing strategies on segmented customers:
# Best Customers- No price incentives, new products, and loyalty programs.
# Big Spenders- Market your most expensive products.
# Almost Lost- Aggresive price incentives
# Lost Customers-Don’t spend too much trying to re-acquire them.

print("Best Customers: ",len(segmented_rfm[segmented_rfm['RFM_Segment']=='111']))
print('Loyal Customers: ',len(segmented_rfm[segmented_rfm['F_quartile']==1]))
print("Big Spenders: ",len(segmented_rfm[segmented_rfm['M_quartile']==1]))
print('Almost Lost: ', len(segmented_rfm[segmented_rfm['RFM_Segment']=='134']))
print('Lost Customers: ',len(segmented_rfm[segmented_rfm['RFM_Segment']=='344']))
print('Lost Cheap Customers: ',len(segmented_rfm[segmented_rfm['RFM_Segment']=='444']))

# K-means gives the best result under the following conditions:

# Data’s distribution is not skewed.
# Data is standardised (i.e. mean of 0 and standard deviation of 1).

# Plot all 3 graphs together for summary findings
def check_skew(df_skew, column,ax):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    ax.title('Distribution of ' + column)
    sns.displot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return
plt.figure(figsize=(9, 9))

fig,(ax1,ax2,ax3) = plt.subplot(nrows=3, ncols=1)
check_skew(RFM_Table,'Recency',ax1)

check_skew(RFM_Table,'Frequency',ax2)

check_skew(RFM_Table,'Monetary_Value',ax3)
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)

# Kmeans 
# from scipy.spatial.distance import cdist
# distortions = [] 
# inertias = [] 
# mapping1 = {} 
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(RFM_Table_scaled) 
#     kmeanModel.fit(RFM_Table_scaled)     
      
#     distortions.append(sum(np.min(cdist(RFM_Table_scaled, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / RFM_Table_scaled.shape[0]) 
#     inertias.append(kmeanModel.inertia_) 
  
#     mapping1[k] = sum(np.min(cdist(RFM_Table_scaled, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) / RFM_Table_scaled.shape[0] 
#     mapping2[k] = kmeanModel.inertia_ 
