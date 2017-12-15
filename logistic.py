from __future__ import division
from util import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score
import operator
import pandas as pd


# Settings
#------------------------------------------------------------------------------------------------------
task = 2 #1 = Red vs. White, 2 = Variety
featurizer   = extractWordFeatures(1)
  #Other options: genericExtractor  extractWordFeatures(1)  extractCharFeatures(5)
  #               featureExtractor([extractWordFeatures(1),extractWordFeatures(2)])
  #               extracWord2VecFeatures(trainWord2Vec(min_count=2,size=400,window=4))
balanceDataset = True

# Import Data
#------------------------------------------------------------------------------------------------------
data_test = pd.read_csv('data-processed/data.test')
data_dev = pd.read_csv('data-processed/data.dev')
data_train = pd.read_csv('data-processed/data.train')


# Prepare Data
#------------------------------------------------------------------------------------------------------
# Get Design Matrices and Feature List
X_train,X_dev,X_test,feature_names = DesignMatrix(data_train=data_train['description'].as_matrix(),
                                                   data_dev=data_dev['description'].as_matrix(),
                                                   data_test=data_test['description'].as_matrix(),
                                                   featurizer=featurizer)

# Get Response Vector
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

#Get Majority Class (from training set)
y_majority = max(collections.Counter(y_train).iteritems(),key=operator.itemgetter(1))[0]


# Models
#------------------------------------------------------------------------------------------------------
if balanceDataset:
    class_weight = 'balanced'
else:
    class_weight = None

#Tune regularization parameter
if False:
    C = [100000,10000,1000,500,100,50,10,5,3,2,1,0.75,0.5,0.1,0.01] #larger values -> less regularization -> more complex model
    regularizationData = []
    for c in C:
        model = LogisticRegression(penalty='l2',C=c,class_weight=class_weight,multi_class='multinomial',solver='sag')
        model = model.fit(X_train,y_train)
        regularizationData.append([np.log10(c),model.score(X_train,y_train),model.score(X_dev,y_dev)])
    regularizationData_np = np.array(regularizationData)

    #Plot
    plt.figure()
    plt.plot(regularizationData_np[:,0],regularizationData_np[:,1],marker='.',label='Train Accuracy')
    plt.plot(regularizationData_np[:,0],regularizationData_np[:,2],marker='.',label='Dev Accuracy')
    plt.ylim((0.45,1))
    plt.legend()
    plt.xlabel('Model Complexity ($-\log(\lambda)$)')
    plt.ylabel('Classification Accuracy')
    plt.savefig('plots/logisticL2Regularization.png')
    plt.show()

#Regularization Parameter
bestC = 1

#Fit Best Model
model = LogisticRegression(penalty='l2',C=bestC,class_weight=class_weight,multi_class='multinomial',solver='sag',max_iter=200)
model = model.fit(X_train,y_train)


# Results
#------------------------------------------------------------------------------------------------------
fout = open('results/task_{0}_logistic_{1}.txt'.format(task,featurizer.func_name),'w')

# Baseline Accuracies (based on majority vote)
null_accuracy_train  = collections.Counter(y_train)[y_majority]/len(y_train)
null_accuracy_dev = collections.Counter(y_dev)[y_majority]/len(y_dev)
null_accuracy_test = collections.Counter(y_test)[y_majority]/len(y_test)
#Null accuracy: accuracy if simply predicting most common class (in training set)
# use of key= and [1] is to get max value in the Counter dictionary
print >>fout, "Baseline Train Accuracy, Baseline Dev Accuracy, Baseline Test\n",null_accuracy_train, null_accuracy_dev, null_accuracy_test

#Full Model Accuracies
train_accuracy = model.score(X_train,y_train)
dev_accuracy   = model.score(X_dev,y_dev)
test_accuracy  = model.score(X_test,y_test)
print >>fout, "Train Accuracy, Dev Accuracy, Test Accuracy\n", train_accuracy, dev_accuracy, test_accuracy

#Full Model F1 Scores
f1_train = f1_score(y_true=y_train,y_pred=model.predict(X_train),average='weighted')
f1_dev = f1_score(y_true=y_dev,y_pred=model.predict(X_dev),average='weighted')
f1_test = f1_score(y_true=y_test,y_pred=model.predict(X_test),average='weighted')
print >>fout, "Train F1 Score, Dev F1 Score, Test F1 Score\n", f1_train, f1_dev, f1_test

fout.close()

# Analysis
#------------------------------------------------------------------------------------------------------
if False:
    # Fitted Parameters
    intercept = model.intercept_
    coef = model.coef_[0,:] #only gets the theta vector for first class in multiclass prediction
                            #use indix other than 0 to get theta vector for other classes

    # Weight Analysis
    weights = pd.DataFrame({'feature': feature_names,'coef': coef})
    weights.sort_values(by='coef',inplace=True,ascending=False)
    outputWeights('analysis/task_{0}_weights_{1}.txt'.format(task,'balanced' if balanceDataset else 'unbalanced'),weights,featureWidth=50)


# Learning Curve
#------------------------------------------------------------------------------------------------------
if False:
    #Build Learning Curve
    def getLearningCurve(class_weight=None):
        model = LogisticRegression(penalty='l2',C=bestC,class_weight=class_weight,multi_class='multinomial',solver='sag',max_iter=500,
                                        warm_start=True)
        trainSetSize = np.arange(start=1000,stop=data_train.shape[0]+1,step=1000)
        learningCurve = []
        for m in trainSetSize:
            # Fit Model
            model = model.fit(X_train[0:m],y_train[0:m])

            # Get Results
            train_accuracy = model.score(X_train[0:m],y_train[0:m])
            dev_accuracy   = model.score(X_dev,y_dev)
            f1 = f1_score(y_dev,model.predict(X_dev),average='weighted')
            learningCurve.append([m,train_accuracy,dev_accuracy,f1])
        
        return np.array(learningCurve)

    #Learning Curves
    learningCurve_unbalanced = getLearningCurve()
    learningCurve_balanced   = getLearningCurve(class_weight='balanced')
    def plotLearningCurve(lCurve,outname='temp'):
        plt.figure()
        plt.plot(lCurve[:,0],lCurve[:,1],label="Train Accuracy",marker=".")
        plt.plot(lCurve[:,0],lCurve[:,2],label="Dev Accuracy",marker=".")
        plt.axhline(y=null_accuracy_dev,label='Baseline Dev Accuracy',c='red',linewidth=0.5)
        plt.legend()
        plt.xlabel('Number of Training Examples')
        plt.ylabel('Classification Accuracy')
        plt.yticks(np.arange(0,1.01,0.1))
        plt.savefig('plots/{0}.png'.format(outname))
    plotLearningCurve(learningCurve_unbalanced,'task_{0}_learningCurve_unbalanced'.format(task))
    plotLearningCurve(learningCurve_balanced,'task_{0}_learningCurve_balanced'.format(task))

    #Compare Balanced vs. Unbalanced F1 Scores
    plt.figure()
    plt.plot(learningCurve_unbalanced[:,0],learningCurve_unbalanced[:,3],label="Dev F1 (unbalanced)",marker=".")
    plt.plot(learningCurve_balanced[:,0],learningCurve_balanced[:,3],label="Dev F1 (balanced)",marker=".")
    plt.legend()
    plt.savefig('plots/task_{0}_learningCurve-compareDev-f1.png'.format(task))


# Confusion Matrix
#------------------------------------------------------------------------------------------------------
if False:
    #Build Confusion Matrix
    def getConfusionMatrix(class_weight=None):
        model = LogisticRegression(penalty='l2',C=bestC,class_weight=class_weight,multi_class='multinomial',solver='sag',max_iter=200)
        model = model.fit(X_train,y_train)
        return confusion_matrix(y_true=y_dev,y_pred=model.predict(X_dev))

    #Confusion Matrix
    cnf_matrix_unbalanced = getConfusionMatrix()
    plot_confusion_matrix(cnf_matrix_unbalanced, classes=labels, vals=cnf_plt_vals,
                          title='Confusion Matrix, Without Normalization',
                          outname='task_{0}_cnf_mat_unnorm_unbalanced'.format(task))
    plot_confusion_matrix(cnf_matrix_unbalanced, classes=labels, vals=cnf_plt_vals, normalize=True,
                          title='Normalized Confusion Matrix',
                          outname='task_{0}_cnf_mat_norm_unbalanced'.format(task))
    cnf_matrix_balanced = getConfusionMatrix(class_weight='balanced')
    plot_confusion_matrix(cnf_matrix_balanced, classes=labels, vals=cnf_plt_vals,
                          title='Confusion Matrix, Without Normalization',
                          outname='task_{0}_cnf_mat_unnorm_balanced'.format(task))
    plot_confusion_matrix(cnf_matrix_balanced, classes=labels, vals=cnf_plt_vals, normalize=True,
                          title='Normalized Confusion Matrix',
                          outname='task_{0}_cnf_mat_norm_balanced'.format(task))


#------------------------------------------------------------------------------------------------------
# End