from __future__ import division
import collections
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords as nltk_stopwords
import scipy.sparse as sp
from sklearn.feature_extraction import DictVectorizer
import itertools
from gensim.models import Word2Vec
from gensim.parsing import PorterStemmer


##########################################################################################################
# Constants | Start
##########################################################################################################
#Note: The first list is a place holder and the second list represents grape varieties. 
#      Modify third list to add words
CENSORED_WORDS = [] + \
    [u'chardonnay',u'pinot',u'noir',u'cabernet',u'sauvignon',u'blanc', 
    u'syrah',u'riesling',u'merlot',u'zinfandel',u'sangiovese',u'malbec',u'tempranillo',
    u'nebbiolo',u'portuguese',u'shiraz',u'corvina',u'rondinella',u'molinara',
    u'barbera',u'gris',u'franc',u'grosso',u'grigio',u'viognier',u'champagne',u'port',u'veltliner'] + \
    ['red','white','cab','sb','grigios','chardonnays','chard','chablis','chards','zin','sauv',
    'cabernets','amarone','bordeaux','verdot','burgandy','burgundian','rioja','sancerre']
CENSORED_TOKEN = '<CENSORED_WORD>'
STOP_WORDS = nltk_stopwords.words('english')
REGEX_CONTRACTIONS = "([a-z]+'[a-z]+)"
REGEX_PUNCTUATION = "([\.\,\?\!\'\"\:\;\-])"

#Maps variety names to a class number (must be unique!) and a grape color
#Only includes varieties with at least 1000 reviews, excluding blends
VARIETY_MAP = {u'Chardonnay':             (0,'White'), 
               u'Pinot Noir':             (1,'Red'),  #champagne is listed separately so Pinot Noir should all be reds
               u'Cabernet Sauvignon':     (2,'Red'), 
               u'Sauvignon Blanc':        (3,'White'), 
               u'Syrah':                  (4,'Red'), 
               u'Riesling':               (5,'White'), 
               u'Merlot':                 (6,'Red'), 
               u'Zinfandel':              (7,'Red'), 
               u'Sangiovese':             (8,'Red'), 
               u'Malbec':                 (9,'Red'), 
               u'Tempranillo':            (10,'Red'),  
               u'Nebbiolo':               (11,'Red'), 
               u'Portuguese Red':         (12,'Red'), 
               u'Shiraz':                 (13,'Red'), 
               u'Corvina, Rondinella, Molinara': (14,'Red'), 
               u'Barbera':                (15,'Red'), 
               u'Pinot Gris':             (16,'White'),  
               u'Cabernet Franc':         (17,'Red'), 
               u'Sangiovese Grosso':      (18,'Red'), 
               u'Pinot Grigio':           (19,'White'),  
               u'Viognier':               (20,'White'), 
               u'Champagne Blend':        (21,'White'), 
               u'Port':                   (22,'Red'), 
               u'Veltliner' :             (23,'White'),
            }
#Note: renamed different Veltliner varieties in importData so no longer need the following (key,val) in COLOR_MAP:
            #'Gr\xc3\xbcner Veltliner': 'White',  #no u'...' here!!!

# Invert the VARIETY_MAP. 
# CLASS_MAP: Class Number - > Variety Name
# Do this programatically to ensure class numbering is consistent
CLASS_MAP = ['Error']*len(VARIETY_MAP.keys())
for key,value in VARIETY_MAP.items():
    CLASS_MAP[value[0]] = key
#Note: to get name of class k : CLASS_MAP[k]
##########################################################################################################
# Constants | End
##########################################################################################################



##########################################################################################################
def importData(path,removeSpecialChars=False,censor=False,mapVariety=False,filter=False,processDescriptions=False,
                processOptions={'removeContractions':False,'removePunctuation':True,
                                            'removeStopWords':True, 'lowerCase':True} ):
    """
    Function to import our dataset from a CSV to a Pandas DataFrame

    @param path: Location of CSV file
    @param censor: If true, removes censored words
    @param mapVariety: If true, creates a 'color' column with vals ('Red','White','N/A'), where
                       'N/A' means we are not considering this variety, and a 
                       'class' column uniquely numbering distinct varieties (with '-1' if we ignore it)
    @param filter: If true, filters data and overrides mapVariety to True
    @param processDescriptions: If true, processes descriptions (e.g. remove stop words)
    @param processOptions: Options for how to process the descritions (when processDescriptions is true)
    """
    # Read CSV
    data = pd.read_csv(path)

    # Rename Column 1
    cols = data.columns.values
    cols[0] = "index"
    data.columns = cols
    
    #Rename 'Veltliner' rows
    data['variety'] = data.apply(lambda row : 'Veltliner' \
                            if row['variety'].find('Veltliner')>=0 else row['variety'], axis=1)
    
    #Remove Special Characters
    if removeSpecialChars:
        regex_string = r'[\x80-\xff]'
        regex_pat = re.compile(regex_string)
        data['description'] = data['description'].str.replace(regex_pat,' ')

        regex_string = "\s+"
        regex_pat = re.compile(regex_string)
        data['description'] = data['description'].str.replace(regex_pat,' ')

    # Censor words from list of fixed words
    if censor:
        if len(CENSORED_WORDS) > 0:
            regex_string = ''
            for w in CENSORED_WORDS:
                regex_string+= r'\b[\S]*' + w + r'[\S]*\b|'
            regex_string = regex_string[0:-1] #remove trailing pipe (|)
            regex_pat = re.compile(regex_string, flags=re.IGNORECASE)
            
            if processDescriptions == True and processOptions['removeStopWords']==True:
                data['description'] = data['description'].str.replace(regex_pat,'')
            else:
                data['description'] = data['description'].str.replace(regex_pat,CENSORED_TOKEN)
    
    #Process the descriptions
    if processDescriptions == True:    
        #Remove Contractions
        if processOptions['removeContractions'] == True:
            regex_pat = re.compile(REGEX_CONTRACTIONS, flags=re.IGNORECASE)
            data['description'] = data['description'].str.replace(regex_pat,'')
    
        #Remove Punctuation
        if processOptions['removePunctuation'] == True:
            regex_pat = re.compile(REGEX_PUNCTUATION)
            data['description'] = data['description'].str.replace(regex_pat,'')

        #Remove Capitalization
        if processOptions['lowerCase'] == True:
            data['description'] = data['description'].str.lower()

        #Remove Stop Words
        if processOptions['removeStopWords'] == True:
            if len(STOP_WORDS) > 0:
                regex_string = ''
                for w in STOP_WORDS:
                    regex_string+=r'\b' + w + r'\b|'
                regex_string = regex_string[0:-1] #remove trailing pipe (|)
                regex_pat = re.compile(regex_string, flags=re.IGNORECASE)
                data['description'] = data['description'].str.replace(regex_pat,'')
        
                #remove extra spaces created by deleting words
                regex_string = "\s+"
                regex_pat = re.compile(regex_string)
                data['description'] = data['description'].str.replace(regex_pat,' ')
    
    #Create new columns with wine color and class number
    mapVariety = filter if filter == True else mapVariety #filter overrides mapVariety because we need the color field to filter data
    if mapVariety:
        data['color'] = data.apply(lambda row : VARIETY_MAP.get(row['variety'],(-1,'N/A'))[1], axis=1)
        data['class'] = data.apply(lambda row : VARIETY_MAP.get(row['variety'],(-1,'N/A'))[0], axis=1)

    #Filter only on selected varieties
    if filter:
        data = data.loc[data['color'] != 'N/A']
        
    return data



##########################################################################################################
word_stemmer = PorterStemmer()

def trainWord2Vec(min_count=2,size=50,window=4):
    """
    Returns a trained word2vec model
    """
    #Load descriptions (i.e. word2vec training data)
    df = importData("data-raw/winemag-data_first150k.csv",removeSpecialChars=True,censor=False,filter=True,processDescriptions=True,
                    processOptions={'removeContractions':False,'removePunctuation':True,'removeStopWords':True,'lowerCase':True}) 
    
    #Convert to an iterable of sentences
    descriptions = df['description'].as_matrix()
    sentences = [[word_stemmer.stem(w) for w in desc.split()] for desc in descriptions] #must be an iterable of iterables

    #Train model
    model = Word2Vec(sentences,min_count=min_count, size=size, window=window)
    return model

def extracWord2VecFeatures(model):
    """
    Returns a function that takes as input a string |text| and returns a dense vector (numpy array)
    that is the featurization of |text| based on the supplied word2vec |model|

    Sums individual word vectors for a given text
    """
    def word2vecFeatures(text):
        result = np.zeros(model.layer1_size)
        for w in text.split():
           w_stem = word_stemmer.stem(w)
           if w_stem in model.wv.vocab.keys():
                result += model[w_stem]
        return result

    return word2vecFeatures



##########################################################################################################
def extractCharFeatures(n,count=True,
        removedCensoredToken=False, removeContractions=False, removePunctuation=False,
        lowerCase=False, removeStopWords=False):
    """
    Returns a function that takes as input a string |text| and returns a sparse vector (dictionary) 
    of n-gram character features.
    Whitespace is removed before forming n-grams

    Input: 
        n: (int) for n-grams (n>=1)

    Optional Args: 
        count: (bool) if true, returns the count of the n-grams
                      if false, returns an indicator for the presence of the n-grams
        removedCensoredToken: (bool) if true, preprocess input to remove presence of CENSORED_TOKEN
        removeContractions: (bool) if true, preprocesses input text to remove contractions 
                                  (i.e. words with apostrophes between text)
        removePunctuation: (bool) if true, preprocesses input text to remove punctuation
        removeStopWords: (bool) if true, prepocesses input text to remove stop words, as per NTLK
        lowerCase: (bool) if true, preprocesses input to place all characters to lower case

    Output: 
        A function. See the output description for the function charFeatures below
    """
    #Clean input errors    
    try:
        n = max(1,int(n)) 
    except:
        n = 1
    
    #Define the function to return
    def charFeatures(text):
        """
        Input: 
            text: (str) the string to process 
        Output: 
            dictionary of extracted character n-grams (key) and their counts/indicator (value), removing whitespace 
        Example (n=3): "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, . . .}
        """
        #Remove CENSORED_TOKEN
        #Note: done separately from removeStopWords because CENSORED_TOKEN might get modified
        #      or glued to other characters by the next steps
        if removedCensoredToken == True:
            text = text.replace(CENSORED_TOKEN,'')

        #Remove Contractions
        if removeContractions == True:
            regex_pat = re.compile(REGEX_CONTRACTIONS, flags=re.IGNORECASE)
            text = re.sub(pattern=regex_pat,repl='',string=text)
    
        #Remove Punctuation
        if removePunctuation == True:
            regex_pat = re.compile(REGEX_PUNCTUATION)
            text = re.sub(pattern=regex_pat,repl='',string=text)

        #Remove Capitalization
        if lowerCase == True:
            text = text.lower()
    
        #Remove Whitespace and Stop Words
        words = text.split()
        if removeStopWords == True:
            text = ''.join([w for w in words if w not in STOP_WORDS])
        else:
            text = ''.join(words)

        #Tokenize                              
        result = collections.defaultdict(int)
        if count == True:
            for i in range(len(text)-n+1):
                result[text[i:i+n]]+=1
        else:
            for i in range(len(text)-n+1):
                result[text[i:i+n]]=1
            
        #Return Value
        return dict(result) #convert back to regular dict type
    
    #Return function
    return charFeatures




##########################################################################################################
def extractWordFeatures(n, count=True, 
                removedCensoredToken=False, removeContractions=False, removePunctuation=False, 
                removeStopWords=False, lowerCase=False):
    """
    Returns a function that takes as input a string |text| and returns a sparse vector (dictionary) 
    of n-gram word features.
    Whitespace is removed before forming n-grams

    Input: 
        n: (int) for n-grams (n>=1)

    Optional Args: 
        count: (bool) if true, returns the count of the n-grams
                      if false, returns an indicator for the presence of the n-grams
        removedCensoredToken: (bool) if true, preprocess input to remove presence of CENSORED_TOKEN
        removeContractions: (bool) if true, preprocesses input text to remove contractions (i.e. apostrophes between text)
        removePunctuation: (bool) if true, preprocesses input text to remove punctuation
        removeStopWords: (bool)if true, prepocesses input text to remove stop words, as per NTLK
        lowerCase: (bool) if true, preprocesses input to place all characters to lower case

    Output: 
        A function. See the output description for the function wordFeatures below
    """
    #Clean input errors    
    try:
        n = max(1,int(n)) 
    except:
        n = 1
 
    #Define the function to return
    def wordFeatures(text):
        """
        Input: 
            text: (str) the string to process 
        Output: 
            dictionary of extracted word n-grams (key) and their counts/indicator (value), removing whitespace 
        Example (n=2): "I like tacos" --> {'I like': 1, 'like tacos': 1}
        """
        #Remove CENSORED_TOKEN
        #Note: done separately from removeStopWords because CENSORED_TOKEN might get modified
        #      or glued to other characters by the next steps
        if removedCensoredToken == True:
            text = text.replace(CENSORED_TOKEN,'')

        #Remove Contractions
        if removeContractions == True:
            regex_pat = re.compile(REGEX_CONTRACTIONS, flags=re.IGNORECASE)
            text = re.sub(pattern=regex_pat,repl='',string=text)
    
        #Remove Punctuation
        if removePunctuation == True:
            regex_pat = re.compile(REGEX_PUNCTUATION)
            text = re.sub(pattern=regex_pat,repl='',string=text)

        #Remove Capitalization
        if lowerCase == True:
            text = text.lower()

        #Split into a word list
        words = text.split()  #before (was causing errors on unicode): word_tokenize(text) #from nltk
                              #should be no difference if feeding in preprocessed text

        #Remove Stop Words
        if removeStopWords == True:
            words = [w for w in words if w not in STOP_WORDS]
    
        #Get Word Counts
        word_ngrams = ngrams(words,n)   
        ngram_counts = None #initialize object name
   
        if count == True:
            ngram_counts = collections.Counter(word_ngrams)
            
        else:
            ngram_counts = collections.defaultdict(int)
            for w in word_ngrams:
                ngram_counts[w] =1
        
        #Return value
        return dict(ngram_counts) #convert back to regular dict type
    
    #Return function
    return wordFeatures



##########################################################################################################
def featureExtractor(individualExtractors):
    """
    Takes as input a list of feature extractor functions,
    each returning a sparse vector, and returns a function 
    that combines their output into a single sparse vector output.
    """
    def extract(x):
        features = collections.defaultdict(float)
        for extractor in individualExtractors:
            extractorName = extractor.func_name
            for key, value in extractor(x).items():
                #append name because there may be identical keys for different extractors
                #note: may still be collision if same extractor used more than once
                features[(extractorName,key)] = value
        return features
    return extract



##########################################################################################################
def DesignMatrix(data_train,data_dev,data_test,featurizer):
    """
    Takes as input a list (or 1-D array) of model input values x^(i) to which to apply a featurizer, phi()
    Produces train/dev sparse matrices which can be used for modelling    

    @param data: 1-dimensional array (list) of inputs
    @param featurizer: a function that generates a dictionary given an item from data
    @param num_train: number of examples to return in X_train. The rest go in X_dev
    """
    if (featurizer.func_name == 'word2vecFeatures'):
        wordVectorSize = featurizer('').shape[0]

        train_feature_mat = np.zeros((data_train.shape[0],wordVectorSize))
        for i,description in enumerate(data_train):
            train_feature_mat[i,:] = featurizer(description)

        dev_feature_mat = np.zeros((data_dev.shape[0],wordVectorSize))
        for i,description in enumerate(data_dev):
            dev_feature_mat[i,:] = featurizer(description)

        test_feature_mat = np.zeros((data_test.shape[0],wordVectorSize))
        for i,description in enumerate(data_test):
            test_feature_mat[i,:] = featurizer(description)

        feature_names = []

        return train_feature_mat,dev_feature_mat,test_feature_mat,feature_names

    else:
        vectorizer = DictVectorizer()
        #doc: -http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html-
            #transform(): Named features not encountered during fit or fit_transform will be silently ignored.

        #Training set - encode sparse features
        feature_list_train = list(map(featurizer,data_train))
        train_feature_mat_dense = vectorizer.fit_transform(feature_list_train).toarray()
        del feature_list_train
        train_feature_mat_sparse = sp.csr_matrix(train_feature_mat_dense)
        del train_feature_mat_dense

        #Dev set - apply sparse features
        feature_list_dev = list(map(featurizer,data_dev))
        dev_feature_mat_dense   = vectorizer.transform(feature_list_dev).toarray()
        del feature_list_dev
        dev_feature_mat_sparse   = sp.csr_matrix(dev_feature_mat_dense)
        del dev_feature_mat_dense

        #Test set - apply sparse features
        feature_list_test = list(map(featurizer,data_test))
        test_feature_mat_dense   = vectorizer.transform(feature_list_test).toarray()
        del feature_list_test
        test_feature_mat_sparse   = sp.csr_matrix(test_feature_mat_dense)
        del test_feature_mat_dense

        #Get feature name encoddings
        feature_names = list(map(str,vectorizer.get_feature_names()))
    
        #Return value
        return train_feature_mat_sparse,dev_feature_mat_sparse,test_feature_mat_sparse,feature_names

    return None


#Used to benchmark speed of custom extractors
def genericExtractor(x):
    return {'numChars':len(x),'numWords':len(x.split())}



##########################################################################################################
def outputWeights(path,weights_df,featureWidth=10):
    """
    Function to save weights to file

    @param path: location to save the file
    @param weights_df: a Pandas data frame with columns 'feature' (str) and 'coef' (convertible to str)
    @param featureWidth: left space padding to be applied to the feature name
    """
    fout = open(path,'w')
    for index, row in weights_df.iterrows():
        print >>fout, row['feature'].rjust(featureWidth) + '\t' + str(row['coef'])
    fout.close()



##########################################################################################################
#Confusion Matrix Plot
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          vals=False,
                          outname='temp'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Source: 
        scikit-learn.org/stable/auto_examples/model_selection/
        plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)
    plt.figure(figsize=(11,8.5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if vals:
        np.set_printoptions(precision=2)
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('plots/{0}.png'.format(outname))



##########################################################################################################
