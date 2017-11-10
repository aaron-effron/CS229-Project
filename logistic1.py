from __future__ import division
from util import *
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp




df = importData("../wine/winemag-data_first150k.csv",censor=True)
df = df.iloc[0:10,:]


map(lambda x: len(x),df['description'].as_matrix())

extractor4 = featureExtractor([extractCharFeatures(3),extractCharFeatures(4)])
print extractor4("I go shopping")


X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([0,0,0,1,1])

d = {'f1':1,'f1':2,'f1':3,'f1':4,'f1':5}

temp = map(extractor4,df['description'].as_matrix())

mat = sp.dok_matrix((4,1), dtype=np.int8)

for key, val in d.items():
    mat[key,1] = val

mat = mat.transpose().tocsr()
print mat.shape



X_train = vectorizer.transform(df['variety'].as_matrix())
y_train = 1*(df['variety'] == 'Cabernet Sauvignon')

model = LogisticRegression()
model = model.fit(X_train,y_train)
model.score(X_train,y_train)
model.predict(X_train)
1-y_train.mean()
model.get_params(deep=True)




#################################################################################################
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import HashingVectorizer
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

y_train, data_train.target
y_test = data_test.target

vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
features1 = extractor4("my text")
features2 = extractor4("his text")
features = np.array([features1,features2])
featureVec = vec.fit_transform(features).toarray()

vec.inverse_transform(featureVec)

X = sp.csr_matrix(featureVec)
y = np.array([0,1])

vec.inverse_transform(X)

vectorizer = HashingVectorizer(analyzer=extractor4)

X_train = vectorizer.transform(data_train.data)
X_test = vectorizer.transform(data_test.data)


model = LogisticRegression()
model = model.fit(X,y)
model.score(X,y)
model.get_params(deep=True)
model.coef_

vec.get_feature_names()



#################################
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
text = df['description'].as_matrix()
features = map(lambda x: extractWordFeatures(x,1),text)
featureVec = vec.fit_transform(features).toarray()

X_train = sp.csr_matrix(featureVec)
y_train = 1*(df['variety'] == 'Cabernet Sauvignon')

model = LogisticRegression()
model = model.fit(X_train,y_train)
model.score(X_train,y_train)

coef = model.coef_.T[:,0]
names = np.array(vec.get_feature_names())[:,0]

np.savetxt("coef.csv",coef)
np.savetxt("names.csv",names)