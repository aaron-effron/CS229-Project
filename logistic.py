from __future__ import division
from util import *
from sklearn.linear_model import LogisticRegression
import operator

#Import Data
#df = importData("winemag-data_first150k.csv",censor=True,filter=True,processDescriptions=True) 
df = importData("winemag-data_first150k_test.csv",censor=True,filter=True,processDescriptions=True) 
     #be patient. It takes about 50 seconds

exit()
dfWV = importData("winemag-data_first150k.csv", filter=True, processDescriptions=True)#, processOptions={'removeContractions':False,'removePunctuation':True,
                                            #'removeStopWords':True, 'lowerCase':True}) 

# Settings
num_examples = 25000
perc_train   = 0.70
featurizer   = extractWordFeatures(1)
  #Other options: genericExtractor  extractWordFeatures(1)  extractCharFeatures(5)
  #               featureExtractor([extractWordFeatures(1),extractWordFeatures(2)])

# Subset Data
data = df.iloc[0:num_examples,:]
dataWV = dfWV.iloc[0:num_examples,:]

num_train = int(num_examples*perc_train)

# Get Design Matrix and Feature List (using time for debugging)
import time
start = time.time()
X_train, X_dev, feature_names = DesignMatrix(data=data['description'].as_matrix(), dataWV=dataWV['description'].as_matrix(),
                                        featurizer=featurizer,
                                        num_train=num_train)

exit()
print(time.time() - start)
#Run-time for 5000 examples
#   genericExtrtactor: 0.062s
#   extractWordFeatures(1) : 5.90s (95x slower)
#   extractCharFeatures(5) : 10.6s (170x slower)
#...after moving preprocessing of text to importData
#   genericExtrtactor: 0.062s
#   extractWordFeatures(1) : 1.26s (20x slower)
#   extractCharFeatures(5) : 9.20s (148x slower)
#       note: this means for loop stepping through string is the bottleneck for charFeatures

# Get Response Vector
#y_train = (1*(data['color'][:num_train] == 'Red')).as_matrix()
#y_dev   = (1*(data['color'][num_train:] == 'Red')).as_matrix()
#y_train = (1*(data['variety'][:num_train] == 'Cabernet Sauvignon')).as_matrix()
#y_dev   = (1*(data['variety'][num_train:] == 'Cabernet Sauvignon')).as_matrix()
y_train = data['class'][:num_train].as_matrix()
y_dev = data['class'][num_train:].as_matrix()

# Fit Model
model = LogisticRegression(penalty='l2')
model = model.fit(X_train,y_train)

# Get Results
train_accuracy = model.score(X_train,y_train)
dev_accuracy   = model.score(X_dev,y_dev)
null_accuracy  = max(collections.Counter(y_dev).iteritems(),key=operator.itemgetter(1))[1]/len(y_dev)
#Null accuracy: accuracy if simply predicting most common class
# use of key= and [1] is to get max value in the Counter dictionary
print "Train Accuracy, Dev Accuracy, Null Accuracy \n", train_accuracy, dev_accuracy, null_accuracy
    

# Fitted Parameters
intercept = model.intercept_
coef = model.coef_[0,:] #only gets the theta vector for first class in multiclass prediction
                        #use indiced other than 0 to get theta vector for other classes

#Predictions
pred = model.predict(X_dev)

# Weight Analysis
'''
weights = pd.DataFrame({'feature': feature_names,'coef': coef})
weights.sort_values(by='coef',inplace=True,ascending=False)
outputWeights('weights.txt',weights,featureWidth=50)
'''