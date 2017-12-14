from util import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Prepare Data
#------------------------------------------------------------------------------------------------------
#Import Processed Data
data_test = pd.read_csv('data-processed/data.test')
data_dev = pd.read_csv('data-processed/data.dev')
data_train = pd.read_csv('data-processed/data.train')

#Featurize
if False:
    #Calcuate
    X_train,X_dev,X_test,feature_names = DesignMatrix(data_train=data_train['description'].as_matrix(),
                                                       data_dev=data_dev['description'].as_matrix(),
                                                       data_test=data_test['description'].as_matrix(),
                                                       featurizer=extracWord2VecFeatures(trainWord2Vec(min_count=2,size=400,window=4)))
    #Save to file
    X_train.tofile('data-processed/w2vFeatures.train')
    X_dev.tofile('data-processed/w2vFeatures.dev')
    X_test.tofile('data-processed/w2vFeatures.test')
    backup = copy.deepcopy(X)

else:
    #Load from file
    X_train = np.fromfile('data-processed/w2vFeatures.train').reshape((56691, 400))
    X_dev = np.fromfile('data-processed/w2vFeatures.dev').reshape((20000, 400))
    X_test = np.fromfile('data-processed/w2vFeatures.test').reshape((20000, 400)) 

#Combine
X = np.concatenate((X_train,X_dev,X_test))
I = np.concatenate((data_train['index'].as_matrix(),data_dev['index'].as_matrix(),data_test['index'].as_matrix()))
    #I[j] gives the index in the original dataset for the jth row of X



# Visualize Grape Varieties Using PCA
#------------------------------------------------------------------------------------------------------
#Compute Reduced Dimension Matrix
pca = PCA(n_components=2,copy=True)
newX = pca.fit_transform(X)
fout = open('results/w2v-pca.txt','w')
fout.write("PCA Explained Variance Ratios: {0}, Total: {1}".format(pca.explained_variance_ratio_,sum(pca.explained_variance_ratio_)))
fout.close() 

#Lookup feature vector for each example
vec_dict = {I[i]:vec for i,vec in enumerate(newX)} #dictionary mapping example index to reduced-dimension feature vector
data = pd.concat((data_train,data_dev,data_test))
data['variety'] = data.apply(lambda row : CLASS_MAP[row['class']], axis=1)
data['pc1'] = data.apply(lambda row : vec_dict[row['index']][0], axis=1)
data['pc2'] = data.apply(lambda row : vec_dict[row['index']][1], axis=1)

#Average vectors across grape varieties
varietyVectors = data[['variety','pc1','pc2']].groupby(['variety']).mean()

#Plot
label_position = {'Shiraz':(-10,-20),'Cabernet Sauvignon':(50,15),'Malbec':(-10,-20),'Sangiovese':(100,-20),'Barbera':(-30,-50),
                  'Sauvignon Blanc':(30,-40),'Nebbiolo':(30,20),'Pinot Gris':(70,-10),'Viognier':(70,-20),'Riesling':(50,10),
                  'Pinot Grigio':(30,20),'Portuguese Red':(-20,5)}
plt.figure(figsize=(15,8))
plt.scatter(varietyVectors['pc1'].as_matrix(),varietyVectors['pc2'].as_matrix())
for label, x, y in zip(varietyVectors.index, varietyVectors['pc1'].as_matrix(), varietyVectors['pc2'].as_matrix()):
    grape_color = VARIETY_MAP[label][1]
    color = 'red' if grape_color == 'Red' else 'yellow'
    xtxt,ytxt = label_position.get(label,(-10,10))
    plt.annotate(
        label,
        xy=(x, y), xytext=(xtxt,ytxt),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('plots/variety_map.png')
plt.show()



# Make Recommendation Based on Cosine Similarity of word2Vec Features
#------------------------------------------------------------------------------------------------------
#Import Raw Data for Lookup
df = importData("data-raw/winemag-data_first150k.csv",filter=True)

#Settings
sample_index = 95130 #95130 92115
topK = 5

#Calculate Cosine Similarity
sample = X[sample_index,:]
sample_norm = np.sqrt(np.sum(sample**2))
X_norm = np.sqrt(np.sum(X**2,axis=1))
cosines = X.dot(sample)/X_norm/sample_norm
cosines_dict = {I[i]:cosVal for i,cosVal in enumerate(cosines)}
cosines_dict[I[sample_index]] #sanity check: should be 1

#Get best match
df['cosine'] = df.apply(lambda row : cosines_dict[row['index']], axis=1)
df.sort_values(by='cosine',inplace=True,ascending=False)

#Print descriptions of best match
fout = open('results/w2v-recommender.txt','w')
fout.write("Input:\n--------------------\n")
fout.write("Index: {0}\n".format(I[sample_index]))
fout.write("Description: {0}\n".format(df['description'][I[sample_index]]))
fout.write("\nOutput (top {0} recommendations):\n--------------------\n".format(topK))
for i,val in enumerate(df[1:(topK+1)].iterrows()): #0th element is the same as the input
    index,row = val
    print >>fout, "Result {0}".format(i+1)
    print >>fout, "\tCosine Similarity: ", row['cosine']
    print >>fout, "\tIndex: ", index #or: row['index']
    print >>fout, "\tVariety: ", row['variety']
    print >>fout, "\tDescription:", row['description']
    print >>fout, "\n"
fout.close()


#------------------------------------------------------------------------------------------------------
# End