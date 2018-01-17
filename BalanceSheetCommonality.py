# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:20:03 2017

@author: JairoFGR
"""
import pandas as pd
from scipy.stats import zscore
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

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
df_norm = df.apply(zscore)

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
  VectorEuDist = EuDist.flatten(1)
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


## Gaussian Mixture Model Selection
lowest_bic = np.infty
bic = []
n_components_range = range(1, 2)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(dx)
        bic.append(gmm.bic(dx))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm