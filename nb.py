"""
==============================
Gaussian Naive Bayes
==============================

"""

from time import time
import numpy as np

from util import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

#Import Data
df = importData("winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 
     #be patient. It takes about 50 seconds

# Settings
#------------------------------------------------------------------------------------------------------
task = 2 #1 = Red vs. White, 2 = Variety
featurizer   = extractWordFeatures(1)
  #Other options: genericExtractor  extractWordFeatures(1)  extractCharFeatures(5)
  #               featureExtractor([extractWordFeatures(1),extractWordFeatures(2)])
  #               extracWord2VecFeatures(trainWord2Vec(min_count=2,size=400,window=4))
balanceDataset = True

data_test = pd.read_csv('data-processed/data.test')
data_dev = pd.read_csv('data-processed/data.dev')
data_train = pd.read_csv('data-processed/data.train')
import time
start = time.time()

X_train,X_dev,X_test,feature_names = DesignMatrix(data_train=data_train['description'].as_matrix(),
                                                   data_dev=data_dev['description'].as_matrix(),
                                                   data_test=data_test['description'].as_matrix(),
                                                   featurizer=featurizer)
print(time.time() - start)

if task == 1:
    labels = ['White','Red']
    cnf_plt_vals = True
    y_test  = (1*(data_test['color'] == 'Red')).as_matrix()
    y_dev   = (1*(data_dev['color'] == 'Red')).as_matrix()
    y_train = (1*(data_train['color'] == 'Red')).as_matrix()
elif task ==2:
    labels = CLASS_MAP
    cnf_plt_vals = False
    y_test  = data_test['class'].as_matrix()
    y_dev   = data_dev['class'].as_matrix()
    y_train = data_train['class'].as_matrix()
else:
    assert False

model = GaussianNB()

model.fit(X_train.todense(), y_train)
print("Train Score with sklearn and GaussianNB: %f"
      % (model.score(X_train.todense(), y_train)))
print("Dev Score with sklearn and GaussianNB: %f"
      % (model.score(X_dev.todense(), y_dev)))
print("Test Score with sklearn and GaussianNB: %f"
      % (model.score(X_test.todense(), y_test)))