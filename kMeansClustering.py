from __future__ import division
from util import *
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import operator
from sklearn.metrics import confusion_matrix
import numpy as np

#Import Data
df = importData("/Users/aferris/Downloads/Stanford/_CS 229/project/winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 
     #be patient. It takes about 50 seconds

# Settings
num_examples = 25000
perc_train   = 0.70
featurizer   = extractWordFeatures(1)
  #Other options: genericExtractor  extractWordFeatures(1)  extractCharFeatures(5)
  #               featureExtractor([extractWordFeatures(1),extractWordFeatures(2)])


# Subset Data
data = df.iloc[0:num_examples,:]
num_train = int(num_examples*perc_train)

# Get Design Matrix and Feature List (using time for debugging)
import time
start = time.time()
X_train, X_dev, feature_names = DesignMatrix(data=data['description'].as_matrix(),
                                        featurizer=featurizer,
                                        num_train=num_train)

#stuff for determining cluster composition
colors = ['White', 'Red']
colorList = np.copy(data['color'])
varieties = ['Chardonnay','Pinot Noir','Cabernet Sauvignon','Sauvignon Blanc','Syrah','Riesling','Merlot','Zinfandel','Sangiovese','Malbec','Tempranillo','Nebbiolo','Portuguese Red','Shiraz','Corvina, Rondinella, Molinara','Barbera','Pinot Gris','Cabernet Franc','Sangiovese Grosso','Pinot Grigio','Viognier','Champagne Blend','Port','Veltliner']
varietyList = np.copy(data['variety'])

#You can input any Compressed Sparse Row format matrix into the kmeans fn
num_clusters = 50

km = KMeans(n_clusters=num_clusters)
km.fit(X_train)
clusters = km.labels_.tolist()
clustersNP = np.asarray(clusters)

for i in range(num_clusters):
    print "cluster", i, "size: ",clusters.count(i)

plotty = []
for cluster in range(num_clusters):
    indexValues = np.where(clustersNP == cluster)[0]
    percentWhite = len(np.where(colorList[indexValues] == 'White')[0])/len(indexValues)
    print "Cluster", cluster, "percent white:", percentWhite*100
    plotty.append(percentWhite)
    
    
x = []
for i in varieties:
    indexValues = np.where(clustersNP == 40)[0]
    temp = len(np.where(varietyList[indexValues] == i)[0])/len(indexValues)
    x.append(temp)
 
"""
plt.bar(np.arange(len(plotty)), plotty)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.tick_params(axis='both', which='major', labelsize=100)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel('Percent White Wines', fontsize = 'large')
plt.xlabel('Cluster Number', fontsize = 'large')"""


#clusters aren't splitting very evenly, try preprocessing with PCA or tf-idf