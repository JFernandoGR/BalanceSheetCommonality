# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:20:03 2017

@author: JairoFGR

"""

import pip
pip.main(['install','mca'])
import mca
# Import libraries #
import sys
sys.path.insert(0, r'C:\Users\Jairo F Gudiño R\Desktop\Balance Sheet Commonality')
from kmodes.kmodes import KModes

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import DistanceMetric

def gower_distance(X):
    """
    This function expects a pandas dataframe as input
    The data frame is to contain the features along the columns. Based on these features a
    distance matrix will be returned which will contain the pairwise gower distance between the rows
    All variables of object type will be treated as nominal variables and the others will be treated as 
    numeric variables.
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    individual_variable_distances = []
    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]]
        try: feature.dtypes[0]
        except KeyError:
          if np.ptp(feature.values)!=0:
           feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
          else:
           feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature)
        else:
          if feature.dtypes[0]=='O':
           feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))      
          else:
            if np.ptp(feature.values)!=0:
             feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
            else:
             feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature)
#             print('Revisar caso de división por ceros')
#             break
        individual_variable_distances.append(feature_dist)
    return np.array(individual_variable_distances).mean(0) #element=6 error de memoria

def getcities(x):
 return (x[:x.notnull().sum(axis=0)-1].str.cat(sep=' '))

# Read Database: #
xl = pd.ExcelFile('C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/FinalDatabase2.xlsx')
# NUMERICAL VARIABLES #
# Read features information
df = xl.parse(xl.sheet_names[0])
BalanceSheetNames = list(df)[1:]
df = df.fillna(0) #Fill NaNs with zeros
dx = pd.DataFrame(df, columns = BalanceSheetNames)
rows,columns = df.shape
# Descriptive Statistics by Feature
descriptivestats = dx.describe()
correlationMatrix = dx.corr() #Correlation Matrix
covarianceMatrix = df.cov() #Covariance Matrix
# Standarization of Numerical Variables
numerical = pd.DataFrame(StandardScaler().fit_transform(df))
# CATEGORICAL VARIABLES #
# Read Industry Type Information
di = xl.parse(xl.sheet_names[1])
IndustryO = di.iloc[:,2].str.split(' ', expand=True).iloc[:,0].str[0]
City = di.iloc[:,6].str.split('-', expand=True).apply(getcities,axis=1)
categorical = pd.DataFrame([di.iloc[:,5],City,di.iloc[:,3]]).T
df_norm = numerical.join(categorical)
# Read industry classes to get info about categories
dw = xl.parse(xl.sheet_names[3])
industriesnumber = len(dw['Letter'])

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
  GowerDist  = np.array(sorted(gower_distance(df_norm_i),key=sum,reverse=True))
  Distance[0][element] = np.triu(GowerDist, k=0).sum() / rows
  mask = np.zeros_like(GowerDist)
  mask[np.triu_indices_from(mask)] = True
  with sns.axes_style("white"):
   fig, Graph = plt.subplots()
   Graph = sns.heatmap(GowerDist, mask=mask, square=True, xticklabels=False, yticklabels=False)
   Graph.set_title(''.join([dw['Name'][element]]))      
   fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'.pdf']))
# Multiple-Correspondence Analysis #: COMPLETAR
x_dummy = mca.dummy(df_norm)
mca_ben = mca.MCA(x_dummy)
#print(mca_ben.fs_r(1))
#print(mca_ben.L)
# http://nbviewer.jupyter.org/github/esafak/mca/blob/master/docs/mca-BurgundiesExample.ipynb
# General Distance Matrix #
df_norm_total = df_norm.reindex(RearrangedRows)
GowerDist_total  = gower_distance(df_norm_total)
mask = np.zeros_like(GowerDist_total)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(GowerDist_total, mask=mask, square=True, xticklabels=False, yticklabels=False)
  Graph.set_title('Distance Matrix for All Enterprises')      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','TotalDistance','.pdf']))

# Estimation: K-Prototypes #
# Testing Parameters #
K_MAX = 2
centroids = []
KK = range(1,K_MAX+1)
for k in KK:
   km = KModes(n_clusters=k, init='Huang', n_init=5, verbose=1)
   km.fit_predict(df_norm) 
   centroids.append(km.cluster_centroids_)
D_k = [gower_distance_tocentroid(df_norm, cent) for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
totss = sum(gower_distance(df_norm)**2)/df_norm.shape[0]       # The total sum of squares
betweenss = totss - tot_withinss          # The between-cluster sum of squares
# Elbow Curve #
kIdx = 9        # K=10
clr = cm.spectral( np.linspace(0,1,10) ).tolist()
mrk = 'os^p<dvh8>+x.'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(KK, betweenss/totss*100, 'b*-')
ax.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12, 
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained (%)')
plt.title('Elbow for K-Prototypes clustering')
# Final Estimation
km = KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(df_norm)
centroids = km.cluster_centroids_
# Retrieving Internal Parameters #

# Plotting convergence #
