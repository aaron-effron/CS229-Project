from util import *
from sklearn.linear_model import LogisticRegression

#Import Data
df = importData("../wine/winemag-data_first150k.csv",censor=True,filter=True) #be patient. It takes about 10 seconds


# Settings
num_examples = 5000
perc_train   = 0.70
featurizer   = extractWordFeatures(1)  #genericExtractor  #extractWordFeatures(1)  #extractCharFeatures(5)

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
#Run-time for 5000 examples
#   genericExtrtactor: 0.062s
#   extractWordFeatures(1) : 5.90s (95x slower)
#   extractCharFeatures(5) : 10.6s (170x slower)

# Get Response Vector
#y_train = (1*(data['color'][:num_train] == 'Red')).as_matrix()
#y_dev   = (1*(data['color'][num_train:] == 'Red')).as_matrix()
y_train = (1*(data['variety'][:num_train] == 'Cabernet Sauvignon')).as_matrix()
y_dev   = (1*(data['variety'][num_train:] == 'Cabernet Sauvignon')).as_matrix()

# Fit Model
model = LogisticRegression(penalty='l2')
model = model.fit(X_train,y_train)

# Get Results
train_accuracy = model.score(X_train,y_train)
dev_accuracy   = model.score(X_dev,y_dev)
print "Train Accuracy, Dev Accuracy, Null Accuracy \n", train_accuracy, dev_accuracy, max(y_dev.mean(),1-y_dev.mean())
    #Null accuracy: accuracy if simply predicting most common class

# Fitted Parameters
intercept = model.intercept_
coef = model.coef_[0,:]

# Weight Analysis
weights = pd.DataFrame({'feature': feature_names,'coef': coef})
weights.sort_values(by='coef',inplace=True,ascending=False)
outputWeights('weights.txt',weights,featureWidth=30)