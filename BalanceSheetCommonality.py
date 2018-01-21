# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:20:03 2017

@author: JairoFGR

"""

# Import libraries #
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.neighbors import DistanceMetric
# https://github.com/nicodv/kmodes --> K-prototypes
import pip
pip.main(['install','mca'])
import mca

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
        if feature.dtypes == np.float: #Se revisa si el feature es un float o no. Modificación realizada.
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
        else:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        individual_variable_distances.append(feature_dist)
    return np.array(individual_variable_distances).mean(0)

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
df_norm = pd.DataFrame(StandardScaler().fit_transform(df))
# CATEGORICAL VARIABLES #
# Read Industry Type Information
di = xl.parse(xl.sheet_names[1])
IndustryO = di.iloc[:,2].str.split(' ', expand=True).iloc[:,0].str[0]
Departments = di.iloc[:,5]
City = di.iloc[:,6].str.split('-', expand=True)
NaNCells = pd.concat([pd.Series(list(range(rows))),5 - City.isnull().sum(axis=1)], axis=1)
di.iloc[:,2].str.split(' ', expand=True)
# Read industry classes to get info about categories
dw = xl.parse(xl.sheet_names[3])
industriesnumber = len(dw['Letter'])


# Descriptive Analysis #
# Multiple-Correspondence Analysis #: COMPLETAR
mca_ben = mca.MCA(df, ncols=39)
print(mca_ben.fs_r(1))
print(mca_ben.L)
# http://nbviewer.jupyter.org/github/esafak/mca/blob/master/docs/mca-BurgundiesExample.ipynb
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
# Add Distances by City, Department # ENDOGENEIDAD?
 with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(df_norm_i, xticklabels=False, yticklabels=False)
  Graph.set_title(''.join([dw['Name'][element]]))      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'_GeneralFeatures.pdf']))
 rows,columns = df_norm_i.shape
 if rows!=0:
# INCLUIR ESCALA DE CONTRIBUCIÓN EN LOS GRÁFICOS #
  GowerDist  = gower_distance(df_norm_i)
  Distance[0][element] = np.triu(GowerDist, k=0).sum() / rows
  mask = np.zeros_like(GowerDist)
  mask[np.triu_indices_from(mask)] = True
 with sns.axes_style("white"):
  fig, Graph = plt.subplots()
  Graph = sns.heatmap(GowerDist, mask=mask, square=True, xticklabels=False, yticklabels=False)
  Graph.set_title(''.join([dw['Name'][element]]))      
  fig.savefig(''.join(['C:/Users/Jairo F Gudiño R/Desktop/Balance Sheet Commonality/',dw['Name'][element],'.pdf']))

# General Distance Matrix #
df_norm_total = df_norm.reindex(RearrangedRows)
# INCLUIR ESCALA DE CONTRIBUCIÓN EN LOS GRÁFICOS #
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
Calinski_Harabasz = np.zeros(dtype=np.int, shape=100)
for element in range(2, 100):
     Calinski_Harabasz[element] = metrics.calinski_harabaz_score(df_norm, kmeans_model.labels_)

