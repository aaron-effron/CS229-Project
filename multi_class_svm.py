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

from pystruct.models import MultiClassClf
from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM)

#Import Data
df = importData("winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 
     #be patient. It takes about 50 seconds

# Settings
num_examples = 25000
#num_examples = 50000
perc_train   = 0.70
#featurizer   = extractWordFeatures(1)
featurizer   = extractCharFeatures(5)
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

print(time.time() - start)
# we add a constant 1 feature for the bias
print X_train.shape
trainBias = np.ones((X_train.shape[0], 1))
print trainBias.shape
devBias = np.ones((X_dev.shape[0], 1))

trainBias = np.ones((X_train.shape[0], 1))
X_train_bias = np.hstack((trainBias, X_train.todense()))

print X_train_bias.shape
X_dev_bias = np.hstack((devBias, X_dev.todense()))

y_train = data['class'][:num_train].as_matrix()
y_dev = data['class'][num_train:].as_matrix()

model = MultiClassClf(n_features=X_train_bias.shape[1], n_classes=24)
n_slack_svm = NSlackSSVM(model, verbose=50, check_constraints=False, C=0.1,
                         batch_size=100, tol=1e-2)
one_slack_svm = OneSlackSSVM(model, verbose=2, C=.10, tol=.001)
subgradient_svm = SubgradientSSVM(model, C=0.1, learning_rate=0.000001,
                                  max_iter=1000, verbose=0)

# n-slack cutting plane ssvm

'''
n_slack_svm.fit(X_train_bias, y_train)
time_n_slack_svm = time() - start
y_pred = np.hstack((n_slack_svm.predict(X_dev_bias)))
print("Score with pystruct n-slack ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_dev), time_n_slack_svm))

## 1-slack cutting plane ssvm
start = time()
one_slack_svm.fit(X_train_bias, y_train)
time_one_slack_svm = time() - start
y_pred = np.hstack((one_slack_svm.predict(X_dev_bias)))
print("Score with pystruct 1-slack ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_dev), time_one_slack_svm))

#online subgradient ssvm
start = time()
subgradient_svm.fit(X_train_bias, y_train)
time_subgradient_svm = time() - start
y_pred = np.hstack((subgradient_svm.predict(X_dev_bias)))

print("Score with pystruct subgradient ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_dev), time_subgradient_svm))

# the standard one-vs-rest multi-class would probably be as good and faster
# but solving a different model

Train Score with sklearn and libsvm: 0.935657 (took 2.000000 seconds)
Dev Score with sklearn and libsvm: 0.655067


libsvm = LinearSVC(multi_class='crammer_singer', C=.1)
libsvm.fit(X_train_bias, y_train)
print("Train Score with sklearn and libsvm: %f"
      % (libsvm.score(X_train_bias, y_train)))
print("Dev Score with sklearn and libsvm: %f"
      % (libsvm.score(X_dev_bias, y_dev)))

coef = libsvm.coef_[0,:] #only gets the theta vector for first class in multiclass prediction
                        #use indiced other than 0 to get theta vector for other classes

weights = pd.DataFrame({'feature': feature_names,'coef': coef})
weights.sort_values(by='coef',inplace=True,ascending=False)
outputWeights('weights.txt',weights,featureWidth=50)


#Binary

Train Score with sklearn and libsvm: 0.997086
Dev Score with sklearn and libsvm: 0.976800
'''

y_train_bin = (1*(data['color'][:num_train] == 'Red')).as_matrix()
y_dev_bin   = (1*(data['color'][num_train:] == 'Red')).as_matrix()

libsvm = LinearSVC(multi_class='crammer_singer', C=.1)
libsvm.fit(X_train_bias, y_train_bin)
print("Train Score with sklearn and libsvm: %f"
      % (libsvm.score(X_train_bias, y_train_bin)))
print("Dev Score with sklearn and libsvm: %f"
      % (libsvm.score(X_dev_bias, y_dev_bin)))

