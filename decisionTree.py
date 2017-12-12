from __future__ import division
from util import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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
print(time.time() - start)

# Get Response Vector
#y_train = (1*(data['color'][:num_train] == 'Red')).as_matrix()
#y_dev   = (1*(data['color'][num_train:] == 'Red')).as_matrix()
#y_train = (1*(data['variety'][:num_train] == 'Cabernet Sauvignon')).as_matrix()
#y_dev   = (1*(data['variety'][num_train:] == 'Cabernet Sauvignon')).as_matrix()
y_train = data['class'][:num_train].as_matrix()
y_dev = data['class'][num_train:].as_matrix()

"""     
#fit with decision tree
model = tree.DecisionTreeClassifier()
model = model.fit(X_train,y_train)

# Get Results

train_accuracy = model.score(X_train,y_train)
dev_accuracy   = model.score(X_dev,y_dev)
null_accuracy  = max(collections.Counter(y_dev).iteritems(),key=operator.itemgetter(1))[1]/len(y_dev)
#Null accuracy: accuracy if simply predicting most common class
# use of key= and [1] is to get max value in the Counter dictionary
print "Train Accuracy, Dev Accuracy, Null Accuracy \n", train_accuracy, dev_accuracy, null_accuracy
"""

#fit with random forest
"""
numTrees = range(1,22,5)
depths = range(50,260,50)
sampleSplits = range(2,23,5)"""
print 'hi'
numTrees = 21	
depths = 250	
sampleSplits = 12

allDev = []
print "Train Accuracy, Dev Accuracy, Null Accuracy \n",
for numTree in numTrees:
    for depth in depths:
        for sample in sampleSplits:
            model = RandomForestClassifier(random_state=0, n_estimators=numTree, max_depth = depth, min_samples_split=sample)
            model = model.fit(X_train,y_train)
            
            train_accuracy = model.score(X_train,y_train)
            dev_accuracy   = model.score(X_dev,y_dev)
            #print "Train Accuracy, Dev Accuracy, Null Accuracy \n", train_accuracy, dev_accuracy, null_accuracy
            print numTree, depth, sample, train_accuracy, dev_accuracy
            allDev.append(dev_accuracy)

npAllDev = np.asarray(allDev)
npAllDev = npAllDev.reshape((5,5,5))

model = RandomForestClassifier(random_state=0, n_estimators=100, max_depth = 260)
model = model.fit(X_train,y_train)
            
train_accuracy = model.score(X_train,y_train)
dev_accuracy   = model.score(X_dev,y_dev)
print "Train Accuracy, Dev Accuracy, Null Accuracy \n", train_accuracy, dev_accuracy, null_accuracy
            
            
#calculate the confusion matrix (ie false positives and negatives)
y_pred = model.predict(X_dev)
cnf_matrix = confusion_matrix(y_dev, y_pred)
plt.imshow()

#normalized
cnf2 = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
plt.imshow(cnf2, cmap = 'Blues', interpolation='none')



#depths are 200 - 260 w/o restrictions
#max_depth=n/a: 0.991485714286 0.572 0.1924
#max_depth=200: 0.990857142857 0.5776 0.1924
#max_depth=150: 0.989142857143 0.577066666667 0.1924
#max_depth=100:0.975771428571 0.580933333333 0.1924
#max_depth=50:0.847371428571 0.550533333333 0.1924

#min_samples_split=n/a: 0.991485714286 0.572 0.1924
#min_samples_split=5: 0.973942857143 0.594666666667 0.1924
#min_samples_split=10: 0.943771428571 0.589066666667 0.1924
#min_samples_split=15: 0.916742857143 0.587333333333 0.1924

#max_depth = 260, default min_samples_split
#40 trees: 0.999542857143 0.610266666667 0.1924
#100 trees: 0.999828571429 0.620133333333 0.1924

#SVD(50) --> decision tree: 0.14333333333333334, 35% of variance explained