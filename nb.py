"""
==============================
Crammer-Singer Multi-Class SVM
==============================

Comparing different solvers on a standard multi-class SVM problem.
"""

from time import time
import numpy as np

#from sklearn.datasets import fetch_mldata
from util import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

#Import Data
df = importData("winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 
     #be patient. It takes about 50 seconds

# Settings
num_examples = 25000
#num_examples = 50000
perc_train   = 0.70
featurizer   = extractWordFeatures(1)
  #Other options: genericExtractor  extractWordFeatures(1)  extractCharFeatures(5)
  #               featureExtractor([extractWordFeatures(1),extractWordFeatures(2)])

# Subset Data
data = df.iloc[0:num_examples,:]
num_train = int(num_examples*perc_train)

import time
start = time.time()
X_train, X_dev, feature_names = DesignMatrix(data=data['description'].as_matrix(),
                                        featurizer=featurizer,
                                        num_train=num_train)
print(time.time() - start)

y_train = data['class'][:num_train].as_matrix()
y_dev = data['class'][num_train:].as_matrix()

#print(time.time() - start)
# we add a constant 1 feature for the bias
#print X_train.shape
#trainBias = np.ones((X_train.shape[0], 1))
#print trainBias.shape
#devBias = np.ones((X_dev.shape[0], 1))

#trainBias = np.ones((X_train.shape[0], 1))
#X_train_bias = np.hstack((trainBias, X_train.todense()))

#print X_train_bias.shape
#X_dev_bias = np.hstack((devBias, X_dev.todense()))

y_train = data['class'][:num_train].as_matrix()
y_dev = data['class'][num_train:].as_matrix()

model = GaussianNB()

model.fit(X_train.todense(), y_train)
print("Train Score with sklearn and GaussianNB: %f"
      % (model.score(X_train.todense(), y_train)))
print("Dev Score with sklearn and GaussianNB: %f"
      % (model.score(X_dev.todense(), y_dev)))

'''
Train Score with sklearn and libsvm: 0.824686
Dev Score with sklearn and libsvm: 0.430667
'''


#Binary

'''
Train Score with sklearn and libsvm: 0.880229
Dev Score with sklearn and libsvm: 0.790800


y_train_bin = (1*(data['color'][:num_train] == 'Red')).as_matrix()
y_dev_bin   = (1*(data['color'][num_train:] == 'Red')).as_matrix()

model = GaussianNB()

model.fit(X_train_bias, y_train_bin)
print("Train Score with sklearn and libsvm: %f"
      % (model.score(X_train_bias, y_train_bin)))
print("Dev Score with sklearn and libsvm: %f"
      % (model.score(X_dev_bias, y_dev_bin)))
'''
