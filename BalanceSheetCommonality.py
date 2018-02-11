# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:20:03 2017

@author: JairoFGR

"""

# Import libraries #

import pip
pip.main(['install','mca'])
import mca

import sys
sys.path.insert(0, r'C:\Users\Jairo F Gudiño R\Desktop\Balance Sheet Commonality')
from kmodes.kprototypes import KPrototypes

from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import DistanceMetric
import itertools

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
        individual_variable_distances.append(feature_dist)
    # np.array(individual_variable_distances).mean(0)
    return sum(individual_variable_distances) / len(individual_variable_distances)

def Manhattan_Distance(feature,centroid,f,v): 
    if v==1:
      op = [(abs(feature - centroid[i][f].astype(np.float)).values) for i in range(0,centroid.shape[0])]
    else:
      op = [(abs(feature - centroid.iloc[i][f])) for i in range(centroid.shape[0])] 
    return op

def Dice_Distance(feature,centroid,f,v):
    if v==1:
     intersection = np.concatenate([np.array(feature == centroid[i][f]) for i in range(0,centroid.shape[0])],axis=1) 
    else:
     intersection = np.concatenate([(feature == centroid.iloc[i][f]) for i in range(0,centroid.shape[0])],axis=1)
     return (~(1*intersection).astype(bool)).astype(int).T

def gower_distance_tocentroid(X,centroid,v):
    # X: Dataframe, centroid: str1344
    if v==1:
     centroid = np.concatenate((centroid[0], centroid[1]), axis=1) #str1344
    individual_variable_distances = []
    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]] #Dataframe float64. feature.values: numpy.ndarray. Para categóricoas dtype('O')
        if v==1: 
         try: feature.dtypes[0]
         except KeyError:
           if np.ptp(feature.values)!=0:
              feature_dist = Manhattan_Distance(feature,centroid,i,1)/ np.ptp(feature.values)
           else:
              feature_dist = Manhattan_Distance(feature,centroid,i,1)
           feature_dist = np.reshape(feature_dist,(feature_dist.shape[0],feature_dist.shape[1])) #float64
         else:
           if feature.dtypes[0]=='O':
            feature_dist = Dice_Distance(feature,centroid,i,1).astype('float64') #float64
           else:
             if np.ptp(feature.values)!=0:
              feature_dist = Manhattan_Distance(feature,centroid,i,1)/ np.ptp(feature.values)
             else:
              feature_dist = Manhattan_Distance(feature,centroid,i,1)
           feature_dist = np.reshape(feature_dist,(feature_dist.shape[0],feature_dist.shape[1]))
        else:
         try: feature = np.array(feature,dtype='float64')
         except ValueError:
              feature_dist = Dice_Distance(feature,centroid,i,2).astype('float64')
         else:
            if np.ptp(feature)!=0:
              feature_dist = Manhattan_Distance(feature,centroid,i,2)/ np.ptp(feature)
            else:
              feature_dist = Manhattan_Distance(feature,centroid,i,2)/ np.ptp([0, 1])
            feature_dist = np.reshape(feature_dist,(feature_dist.shape[0],feature_dist.shape[1]))
        individual_variable_distances.append(feature_dist)
    return sum(individual_variable_distances) / len(individual_variable_distances)

def partition_gower_distance(X,n):
# X = df_norm_i
# n = 5
 t = X.T
 _,columns = t.shape
 finalcolumns = columns
 while (finalcolumns % n != 0):
    finalcolumns = finalcolumns - 1
 t_main = t.iloc[:,0:finalcolumns]
 partition =  np.hsplit(t_main,n)
 t_remain = t.iloc[:,finalcolumns:columns]
 _,columnsshape = t_remain.shape
 if columnsshape != 0:
    s = n+1
    partition.append(t_remain)
 else:
    s = n
 possiblecombinations = list(itertools.combinations(np.array(range(s)),2))
 [possiblecombinations.append((i,i)) for i in range(s)]
 possiblecombinations = sorted(possiblecombinations)
 distances = [gower_distance_tocentroid(partition[possiblecombinations[f][0]].T,
       partition[possiblecombinations[f][1]].T,2) for f in range(len(possiblecombinations))]
 positions = pd.DataFrame(possiblecombinations)
 matrix = np.zeros(shape=(t.shape[1],t.shape[1]))
 for element in range(n):  
    mat_dist = np.array(positions[positions[0] == element].index).T
    element*
    test = np.concatenate([distances[f].T for f in mat_dist], axis=1)
    if element==0:
       matrix[0:test.shape[1]][0:test.shape[0]]=test
    else:
       
    
 
 return matrix
    
def getcities(x):
    return (x[:x.notnull().sum(axis=0)-1].str.cat(sep=' '))

# Read Database: #
xl = pd.ExcelFile('C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/FinalDatabase2.xlsx')
# NUMERICAL VARIABLES #
# Read features information
df = xl.parse(xl.sheet_names[0])
BalanceSheetNames = list(df)[0:]
df = df.fillna(0) #Fill NaNs with zeros
rows,columns = df.shape
# Standarization of Numerical Variables
numerical = pd.DataFrame(StandardScaler().fit_transform(df), columns = BalanceSheetNames)# Descriptive Statistics by Feature
dx = pd.DataFrame(df, columns = BalanceSheetNames)
descriptivestats = dx.describe()
correlationMatrix = dx.corr() #Correlation Matrix
covarianceMatrix = df.cov() #Covariance Matrix                       
# CATEGORICAL VARIABLES #
# Read Industry Type Information
di = xl.parse(xl.sheet_names[1])
IndustryO = di.iloc[:,2].str.split(' ', expand=True).iloc[:,0].str[0]
City = di.iloc[:,6].str.split('-', expand=True).apply(getcities,axis=1).rename("Ciudad")
categorical = pd.DataFrame([di.iloc[:,5],City,di.iloc[:,3]]).T
df_norm = numerical.join(categorical)
# Read industry classes to get info about categories
dw = xl.parse(xl.sheet_names[3])
industriesnumber = len(dw['Letter'])

# Descriptive Analysis #
# Mean Values by Feature of Industry
dx['Industry'] = pd.DataFrame(IndustryO)
MeanbyVariable = dx.groupby('Industry').mean()
# Extraction of Gower Distances by Sector & Heatmaps #
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
   fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'.jpg']))
# MCA & PCA Analysis #
F = 20
x_dummy = mca.dummy(df_norm.iloc[:,[-3,-2,-1]])
mca_ben = mca.MCA(x_dummy,ncols=3)
explained_variance = mca_ben.expl_var(greenacre=False, N = F)*100
explained_variance.sum()
MCAcolumns = [("Factor" + str(i+1)) for i in range(F)]
MCAFactorScores = pd.DataFrame(mca_ben.fs_r(N=F), columns = MCAcolumns)
PCADataframe = pd.concat([df_norm.iloc[:,range(df_norm.shape[1]-3)],MCAFactorScores],axis=1)
reduced_data = PCA(n_components=2).fit_transform(PCADataframe)
fig, Graph = plt.subplots()
Graph = plt.scatter(reduced_data[:,0],reduced_data[:,1], color='blue')
plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
plt.title('Similaridad entre Empresas Grandes y Medianas del Sector Real: PCA')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','PCA','.pdf']))

# General Distance Matrix #
df_norm_total = df_norm.reindex(RearrangedRows)
GowerDist_total  = partition_gower_distance(df_norm_total,10)
mask = np.zeros_like(GowerDist_total)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(GowerDist_total, mask=mask, square=True, xticklabels=False, yticklabels=False)
  Graph.set_title('Distance Matrix for All Enterprises')      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','TotalDistance','.jpg']))

# Estimation: K-Prototypes #
# Testing Number of Clusters #
K_MAX = 10
centroids = []
KK = range(1,K_MAX+1)
for k in KK:
   km = KPrototypes(n_clusters=k, init='Huang', n_init=5, verbose=1)
   km.fit_predict(df_norm.values, categorical = [39,40,41]) 
   centroids.append(km.cluster_centroids_)
D_k = [gower_distance_tocentroid(df_norm, cent,1) for cent in centroids]
# axis=0: Horizontal. axis=1: Vertical
cIdx = [np.argmin(D,axis=0) for D in D_k]
dist = [np.min(D,axis=0) for D in D_k]
tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
totss = sum(partition_gower_distance(df_norm,10)**2)/df_norm.shape[0]       # The total sum of squares
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
km = KPrototypes(n_clusters=2, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(df_norm.values, categorical = [39,40,41]) 
centroids = km.cluster_centroids_
# Retrieving Internal Parameters #

# Plotting convergence #
