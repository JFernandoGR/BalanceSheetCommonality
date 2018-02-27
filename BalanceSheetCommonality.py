# -*- coding: utf-8 -*-
"""
@author: Jairo Fernando Gudiño Rosero
"""

# Import libraries #

import pip
pip.main(['install','mca'])
import mca

import sys
sys.path.insert(0, r'C:\Users\Jairo F Gudiño R\Desktop\Balance Sheet Commonality')
from kmodes.kprototypes import KPrototypes

from sklearn.decomposition import IncrementalPCA
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import DistanceMetric
import itertools
import math

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
           feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature)
        else:
          if feature.dtypes[0]=='O':
           feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))      
          else:
           feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature)
        individual_variable_distances.append(feature_dist)
    return sum(individual_variable_distances) / len(individual_variable_distances)

def Manhattan_Distance(feature,centroid,f,v): 
    if v==1:
      op = [(abs(feature - centroid[i][f].astype(np.float)).values) for i in range(0,centroid.shape[0])]
    else:
      op = [(abs(feature - centroid.iloc[i][f])) for i in range(centroid.shape[0])] 
    return np.concatenate(op, axis=1)

def Dice_Distance(feature,centroid,f,v):
    if v==1:
     intersection = np.concatenate([(feature == centroid[i][f]) for i in range(0,centroid.shape[0])],axis=1) 
    else:
     intersection = np.concatenate([(feature == centroid.iloc[i][f]) for i in range(0,centroid.shape[0])],axis=1)
    return np.array(((1*~intersection).astype(bool)).astype(int).T)

def gower_distance_tocentroid(X,centroid,v):
    # X: Dataframe, centroid: str1344
    if v==1:
     centroid = np.concatenate((centroid[0], centroid[1]), axis=1)#str1344
    individual_variable_distances = []
    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]] #Dataframe float64. feature.values: numpy.ndarray. Para categóricoas dtype('O')
        if v==1: 
         try: feature.dtypes[0]
         except KeyError:
              feature_dist = Manhattan_Distance(feature,centroid,i,1)
         else:
           if feature.dtypes[0]=='O':
              feature_dist = Dice_Distance(feature,centroid,i,1).astype('float64').T
           else:
              feature_dist = Manhattan_Distance(feature,centroid,i,1)
        else:
         try: feature = np.array(feature,dtype='float64')
         except ValueError:
              feature_dist = Dice_Distance(feature,centroid,i,2).astype('float64').T
         else:
              feature_dist = Manhattan_Distance(feature,centroid,i,2)
        individual_variable_distances.append(feature_dist)
    return sum(individual_variable_distances) / len(individual_variable_distances)

def partition_gower_distance(X,n):
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
 f1 = 0
 for element in range(n+1):  
    mat_dist = np.array(positions[positions[0] == element].index).T
    phi = np.concatenate([distances[f] for f in mat_dist], axis=1)
    r,c = phi.shape
    row = np.array(range(f1,(f1 + r)),dtype=np.intp)
    col = np.array(range(f1,(f1 + c)),dtype=np.intp)
    matrix[row[:,np.newaxis],col]= phi
    matrix[col[:,np.newaxis],row]= np.array(phi.T)
    f1 = (f1 + r)
 return matrix

def getcities(x):
    return (x[:x.notnull().sum(axis=0)-1].str.cat(sep=' '))

def maximumshare(x):
    labels = x.join(pd.DataFrame(np.ones(shape=(x.shape[0],1)), columns = ['ones']))
    groupsdata = labels['ones'].groupby([labels['names']]).sum()
    return groupsdata.max()/x.shape[0]

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
LenIndex = [[0] *industriesnumber for i in range(1)]
RearrangedRows = np.zeros(dtype=np.int, shape=len(IndustryO))
for element in range(0,industriesnumber):
 Indexes = IndustryO[IndustryO==dw['Letter'][element]].index
 LenIndex[0][element] = len(Indexes)
 if (element!=0 and element<7):
    p = np.where(RearrangedRows == 0)[0][0]
    RearrangedRows[p:p+len(Indexes)] = Indexes
 elif (element!=0 and element>=7):
    p = np.where(RearrangedRows == 0)[0][1]
    RearrangedRows[p:p+len(Indexes)] = Indexes
 else:
    RearrangedRows[0:len(Indexes)] = Indexes
 df_norm_i = df_norm.iloc[Indexes,:]
 rows,columns = df_norm_i.shape
 if rows!=0:
  # Distances among Enterprises by Industry #
  GowerDist  = np.array(sorted(gower_distance(df_norm_i),key=sum,reverse=True))
  Distance[0][element] = np.triu(GowerDist, k=0).sum()/(rows)
  # Visualization of Distances among Enterprises by Industry #
  mask = np.zeros_like(GowerDist)
  mask[np.triu_indices_from(mask)] = True
  with sns.axes_style("white"):
   fig, Graph = plt.subplots()
   Graph = sns.heatmap(GowerDist, mask=mask, square=True, xticklabels=False, yticklabels=False)
   Graph.set_title(''.join([dw['Name'][element]]))      
   fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'.jpg']))

# General Distance Matrix #
df_norm_total = df_norm.reindex(RearrangedRows)
GowerDist_total  = partition_gower_distance(df_norm_total,10)
# Distances between industries #
limits = np.cumsum(LenIndex)
between_distances = np.zeros(shape=(industriesnumber,industriesnumber))
f = 0
for i in range(industriesnumber):
    partmatrix = GowerDist_total[:,np.array(range(f,(limits[i])),dtype=np.intp)]
    w = 0
    for s in range(industriesnumber):
        n = partmatrix[np.array(range(w,(limits[s])),dtype=np.intp),:]
        between_distances[s,i] = n.sum()/math.sqrt(n.shape[0]*n.shape[1])
        w = limits[s]
    f = limits[i]
between_distances2 = np.tril(between_distances)
fig, Graph = plt.subplots(figsize=(18,18))
Graph = sns.heatmap(between_distances, annot=True, fmt=".2f",square=True)
Graph.set_yticklabels(dw['Letter'].sort_values(ascending=False),minor=False)
Graph.set_xticklabels(dw['Letter'],minor=False)
plt.title('Similarity among Colombian Economic Sectors')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','IndustriesSimilarity','.pdf']))
pd.DataFrame(between_distances).to_csv('C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/IndustryDistances.csv', sep=',', encoding='utf-8', index=True)
#heatmap = Graph.pcolor(between_distances, cmap=plt.cm.Blues)
#plt.colorbar(heatmap)
#Graph.invert_yaxis()
#Graph.set_yticklabels(dw['Letter'],minor=False)
#Graph.set_xticklabels(dw['Letter'],minor=False)
#plt.title('Similarity among Colombian Economic Sectors')
#plt.show()
#fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','IndustriesSimilarity','.pdf']))
pd.DataFrame(RearrangedRows).to_csv('C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/Order.csv', sep=',', encoding='utf-8', index=True)

# Estimation: K-Prototypes #
# Testing Number of Clusters #
K_MAX = 29
centroids_huang = []
centroids_cao = []
labels_huang = []
labels_cao = []
gamma_huang = []
gamma_cao = []
KK = range(1,K_MAX+1)
for k in KK:
   km = KPrototypes(n_clusters=k, init='Huang', n_init=10, verbose=1)
   km.fit_predict(df_norm.values, categorical = [39,40,41]) 
   centroids_huang.append(km.cluster_centroids_)
   labels_huang.append(km.labels_)
   gamma_huang.append(km.gamma)
   km = KPrototypes(n_clusters=k, init='Cao', n_init=10, verbose=1)
   km.fit_predict(df_norm.values, categorical = [39,40,41]) 
   centroids_cao.append(km.cluster_centroids_)
   labels_cao.append(km.labels_)
   gamma_cao.append(km.gamma)
D_k_huang = [gower_distance_tocentroid(df_norm,cent,1) for cent in centroids_huang]
D_k_cao = [gower_distance_tocentroid(df_norm,cent,1) for cent in centroids_cao]
# axis=0: Horizontal. axis=1: Vertical
dist_huang = [np.min(D,axis=1) for D in D_k_huang]
tot_withinss_huang = [sum(d**2) for d in dist_huang]  # Total within-cluster sum of squares
dist_cao = [np.min(D,axis=1) for D in D_k_cao]
tot_withinss_cao = [sum(d**2) for d in dist_cao]  # Total within-cluster sum of squares
GowerDist_m = np.tril(partition_gower_distance(df_norm,10)) # GowerDist_m = np.tril(GowerDist_total)
totss = sum(sum(GowerDist_m**2))/df_norm.shape[0] # The total sum of squares
betweenss_huang = totss - tot_withinss_huang # The between-cluster sum of squares
betweenss_cao = totss - tot_withinss_cao # The between-cluster sum of squares
# Elbow Curve #
kIdx = 28
fig = plt.figure()
huang = plt.plot(KK, betweenss_huang/totss*100, 'b*-',label='Huang Inizialitation')
cao = plt.plot(KK, betweenss_cao/totss*100, 'r*-',label='Cao Inizialitation')
plt.legend()
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained (%)')
plt.title('Elbow Curve for K-Prototypes Clustering')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','ElbowCurve','.pdf']))
# Cluster Concentration through Number of Clusters #
percentages_cao = pd.Series([maximumshare(pd.DataFrame(cent, columns=['names'])) for cent in labels_cao[0:]])
percentages_huang = pd.Series([maximumshare(pd.DataFrame(cent, columns=['names'])) for cent in labels_huang[0:]])
fig = plt.figure()
huang = plt.plot(KK, percentages_cao, 'b*-',label='Huang Inizialitation')
cao = plt.plot(KK, percentages_huang, 'r*-',label='Cao Inizialitation')
plt.legend()
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of enterprises in the largest cluster (%)')
plt.title('Percentage of enterprises in the largest cluster')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','LargestCluster','.pdf']))

# Additional Analysis #
# MCA & PCA Analysis #
F = 20
x_dummy = mca.dummy(df_norm.iloc[:,[-3,-2,-1]])
mca_ben = mca.MCA(x_dummy,ncols=3)
explained_variance = mca_ben.expl_var(greenacre=False, N = F)*100
explained_variance.sum()

# MCA Explained Variance #
MCAcolumns = [("F" + str(i+1)) for i in range(F)]
fig, Graph = plt.subplots()
Graph = plt.bar(np.arange(len(MCAcolumns)),explained_variance, align='center', alpha=0.5)
plt.xticks(np.arange(len(MCAcolumns)), MCAcolumns)
plt.ylabel('Percentage')
plt.title('Explained Variance by Factor (%): Multiple Correspondence Analysis')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','MCA','.pdf']))
ft = mca_ben.fs_r(N=F)
# PCA Explained Variance #
MCAFactorScores = pd.DataFrame(ft, columns = MCAcolumns)
PCADataframe = pd.concat([df_norm.iloc[:,range(df_norm.shape[1]-3)],MCAFactorScores],axis=1)
PCAModel = IncrementalPCA(n_components=3)
reduced_data = PCAModel.fit_transform(PCADataframe)
explained_variancePCA = PCAModel.explained_variance_ratio_*100
PCAcolumns = [("F" + str(i+1)) for i in range(3)]
fig, Graph = plt.subplots()
Graph = plt.bar(np.arange(len(PCAcolumns)),explained_variancePCA, align='center', alpha=0.5)
plt.xticks(np.arange(len(PCAcolumns)), PCAcolumns)
plt.ylabel('Percentage')
plt.title('Explained Variance by Factor (%): Principal Component Analysis')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','PCA','.pdf']))
explained_variancecumPCA = explained_variancePCA.cumsum()
# Similarity 3D Plot#
fig, Graph = plt.subplots()
ax = Axes3D(fig)
Graph = ax.scatter(reduced_data[:,0],reduced_data[:,1],reduced_data[:,2], color='red')
plt.xlabel('PCA-1')
plt.ylabel('PCA-2')
ax.set_zlabel('PCA-3')
plt.title('Similarity among Colombian Real Sector Enterprises: PCA Factors')
plt.show()
fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/','Similarity','.pdf']))

####### END OF THE CODE #######
