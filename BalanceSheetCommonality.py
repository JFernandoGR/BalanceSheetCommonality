# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:20:03 2017

@author: JairoFGR
"""
# Import libraries #
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Read Database: #
xl = pd.ExcelFile('C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/FinalDatabase2.xlsx')
# Read features information
df = xl.parse(xl.sheet_names[0])
BalanceSheetNames = list(df)[1:]
df = df.fillna(0) #Fill NaNs with zeros
dx = pd.DataFrame(df, columns = BalanceSheetNames)
# Descriptive Statistics by Feature
descriptivestats = dx.describe()
correlationMatrix = dx.corr() #Correlation Matrix
covarianceMatrix = df.cov() #Covariance Matrix
# Read Industry Type Information
di = xl.parse(xl.sheet_names[1])
IndustryO = di.iloc[:,2].str.split(' ', expand=True).iloc[:,0].str[0]
# Read Industry Classes & Replace Categories by Numbers
dw = xl.parse(xl.sheet_names[3])
industriesnumber = len(dw['Letter'])
# Standarization of Variables
df_norm = pd.DataFrame(StandardScaler().fit_transform(df))

# Descriptive Analysis #
# Mean Values by Feature of Industry
dx['Industry'] = pd.DataFrame(IndustryO)
MeanbyVariable = dx.groupby('Industry').mean()
# Extraction of Euclidean Distances by Sector & Heatmaps #
Distance = [[0] *industriesnumber for i in range(1)]
RearrangedRows = np.zeros(dtype=np.int, shape=len(IndustryO))
for element in range(0,industriesnumber):
 Indexes = IndustryO[IndustryO==dw['Letter'][element]].index
 if (element!=0 and element<7):
    p = np.where(RearrangedRows == 0)[0][0]
    RearrangedRows[p:p+len(Indexes)] = Indexes
 elif (element!=0 and element>=7):
    p = np.where(RearrangedRows == 0)[0][1]
    RearrangedRows[p:p+len(Indexes)] = Indexes
 else:
    RearrangedRows[0:len(Indexes)] = Indexes
 df_norm_i = df_norm.iloc[Indexes,:]
# General Features by Industry #
 with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(df_norm_i, xticklabels=False, yticklabels=False)
  Graph.set_title(''.join([dw['Name'][element]]))      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'_GeneralFeatures.pdf']))
 rows,columns = df_norm_i.shape
 if rows!=0:
  EuDist  = euclidean_distances(df_norm_i, df_norm_i)
  Distance[0][element] = np.triu(EuDist, k=0).sum() / rows
  mask = np.zeros_like(EuDist)
  mask[np.triu_indices_from(mask)] = True
 with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(EuDist, mask=mask, square=True, xticklabels=False, yticklabels=False)
  Graph.set_title(''.join([dw['Name'][element]]))      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'.pdf']))

# General Distance Matrix #
df_norm_total = df_norm.reindex(RearrangedRows)
EuDist_total  = euclidean_distances(df_norm_total, df_norm_total)
mask = np.zeros_like(EuDist_total)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(EuDist_total, mask=mask, square=True, xticklabels=False, yticklabels=False)
  Graph.set_title('Distance Matrix for All Enterprises')      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','TotalDistance','.pdf']))

# Estimation #
db = DBSCAN(eps=20, min_samples=1).fit(df_norm)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print("Calinski-Harabasz Coefficient: %0.3f"
      % metrics.calinski_harabaz_score(df_norm, labels))
