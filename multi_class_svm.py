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
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC

from pystruct.models import MultiClassClf
from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM, FrankWolfeSSVM)

#Import Data
df = importData("../wine/winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 
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

import time
start = time.time()
X_train, X_dev, feature_names = DesignMatrix(data=data['description'].as_matrix(),
                                        featurizer=featurizer,
                                        num_train=num_train)
print(time.time() - start)

y_train = data['class'][:num_train].as_matrix()
y_dev = data['class'][num_train:].as_matrix()


# we add a constant 1 feature for the bias
X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_dev_bias = np.hstack([X_test, np.ones((X_dev.shape[0], 1))])


model = MultiClassClf(n_features=X_train_bias.shape[1], n_classes=10)
n_slack_svm = NSlackSSVM(model, verbose=2, check_constraints=False, C=0.1,
                         batch_size=100, tol=1e-2)
one_slack_svm = OneSlackSSVM(model, verbose=2, C=.10, tol=.001)
subgradient_svm = SubgradientSSVM(model, C=0.1, learning_rate=0.000001,
                                  max_iter=1000, verbose=0)

fw_bc_svm = FrankWolfeSSVM(model, C=.1, max_iter=50)
fw_batch_svm = FrankWolfeSSVM(model, C=.1, max_iter=50, batch_mode=True)

# n-slack cutting plane ssvm
start = time()
n_slack_svm.fit(X_train_bias, y_train)
time_n_slack_svm = time() - start
y_pred = np.hstack(n_slack_svm.predict(X_test_bias))
print("Score with pystruct n-slack ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_n_slack_svm))

## 1-slack cutting plane ssvm
start = time()
one_slack_svm.fit(X_train_bias, y_train)
time_one_slack_svm = time() - start
y_pred = np.hstack(one_slack_svm.predict(X_test_bias))
print("Score with pystruct 1-slack ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_one_slack_svm))

#online subgradient ssvm
start = time()
subgradient_svm.fit(X_train_bias, y_train)
time_subgradient_svm = time() - start
y_pred = np.hstack(subgradient_svm.predict(X_test_bias))

print("Score with pystruct subgradient ssvm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_subgradient_svm))

# the standard one-vs-rest multi-class would probably be as good and faster
# but solving a different model
libsvm = LinearSVC(multi_class='crammer_singer', C=.1)
start = time()
libsvm.fit(X_train, y_train)
time_libsvm = time() - start
print("Score with sklearn and libsvm: %f (took %f seconds)"
      % (libsvm.score(X_test, y_test), time_libsvm))


start = time()
fw_bc_svm.fit(X_train_bias, y_train)
y_pred = np.hstack(fw_bc_svm.predict(X_test_bias))
time_fw_bc_svm = time() - start
print("Score with pystruct frankwolfe block coordinate ssvm: %f (took %f seconds)" %
      (np.mean(y_pred == y_test), time_fw_bc_svm))

start = time()
fw_batch_svm.fit(X_train_bias, y_train)
y_pred = np.hstack(fw_batch_svm.predict(X_test_bias))
time_fw_batch_svm = time() - start
print("Score with pystruct frankwolfe batch ssvm: %f (took %f seconds)" %
      (np.mean(y_pred == y_test), time_fw_batch_svm))
'''