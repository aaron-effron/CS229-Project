from __future__ import division
from util import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import operator
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import TruncatedSVD

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
print(time.time() - start)

#process data with SVD (ie latent semantic analysis) becasue PCA doesn't work on compressed matrices

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
svd.fit(X_train.T) #this reduces the number of words not the number of examples
X_train_reduced = svd.components_.T #use this to plug back into your model of choice