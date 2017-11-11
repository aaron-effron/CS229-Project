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
from multiprocessing import Pool


##########################################################################################################
#Note: The first list is a place holder and the second list represents grape varieties. 
#      Modify third list to add words
CENSORED_WORDS = [] + \
    [u'chardonnay',u'pinot',u'noir',u'cabernet',u'sauvignon',u'blanc', 
    u'syrah',u'riesling',u'merlot',u'zinfandel',u'sangiovese',u'malbec',u'tempranillo',
    u'nebbiolo',u'portuguese',u'shiraz',u'corvina',u'rondinella',u'molinara',
    u'barbera',u'gris',u'franc',u'grosso',u'grigio',u'viognier',u'champagne',u'port',u'veltliner'] + \
    ['red','white','cab']
CENSORED_TOKEN = '<CENSORED_WORD>'
PUNCTUATION_LIST = ['.',',','?','!',"'",'"',':',';','-']
STOP_WORDS = nltk_stopwords.words('english')

#Only includes varieties with at least 1000 reviews, excluding blends
COLOR_MAP = {u'Chardonnay': 'White', 
             u'Pinot Noir': 'Red',  #champagne is listed separately so Pinot Noir should all be reds
             u'Cabernet Sauvignon': 'Red', 
             u'Sauvignon Blanc': 'White', 
             u'Syrah': 'Red', 
             u'Riesling': 'White', 
             u'Merlot': 'Red', 
             u'Zinfandel': 'Red', 
             u'Sangiovese': 'Red', 
             u'Malbec': 'Red', 
             u'Tempranillo': 'Red',  
             u'Nebbiolo': 'Red', 
             u'Portuguese Red': 'Red', 
             u'Shiraz': 'Red', 
             u'Corvina, Rondinella, Molinara': 'Red', 
             u'Barbera': 'Red', 
             u'Pinot Gris': 'White',  
             u'Cabernet Franc': 'Red', 
             u'Sangiovese Grosso': 'Red', 
             u'Pinot Grigio': 'White',  
             u'Viognier': 'White', 
             u'Champagne Blend': 'White', 
             u'Port': 'Red', 
             u'Veltliner' : 'White',
            }
#Note: renamed different Veltliner varieties in importData so no longer need the following (key,val) in COLOR_MAP:
            #'Gr\xc3\xbcner Veltliner': 'White',  #no u'...' here!!!



##########################################################################################################
def importData(path,getColor=False,censor=False,filter=False):
    """
    Function to import our dataset from a CSV to a Pandas DataFrame

    @param path: Location of CSV file
    @param getColor: If true, creates a 'color' column with vals ('Red','White','N/A')
                     'N/A' means we are not considering this variety
    @param censor: If true, removes censored words
    @param filter: If true, filters data and overrides getColor to True
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
    
    # Censor words from list of fixed words
    if censor:
        if len(CENSORED_WORDS) > 0:
            regex_string = ''
            for w in CENSORED_WORDS:
                regex_string+=r'\b' + w + r'\b|'
            regex_string = regex_string[0:-1] #remove trailing pipe (|)
            regex_pat = re.compile(regex_string, flags=re.IGNORECASE)

            data['description'] = data['description'].str.replace(regex_pat,CENSORED_TOKEN)
    
    def applyColor():
        #Create new column with wine color
        data['color'] = data.apply(lambda row : COLOR_MAP.get(row['variety'],'N/A'), axis=1)

    if getColor or filter:
        applyColor()

    if filter:
        data = data.loc[data['color'] != 'N/A']
        
    return data


#Test Cases:
if False:
    df = importData("../wine/winemag-data_first150k.csv",censor=True)
    df2 = importData("../wine/winemag-data_first150k.csv",censor=True,filter=True)

    x = []
    vals = set(df2['variety'].values)
    len(vals)
    for v in vals:
        if v.find('Veltliner') != -1:
            x.append(v)



##########################################################################################################
def extractCharFeatures(n,
        count=True,removeContractions=False,removePunctuation=True,
        removeStopWords=True, lowerCase=True):
    """
    Returns a function that takes as input a string |text| and returns a sparse vector (dictionary) 
    of n-gram character features.
    Whitespace is removed before forming n-grams

    Input: 
        n: (int) for n-grams (n>=1)

    Optional Args: 
        count: (bool) if true, returns the count of the n-grams
                      if false, returns an indicator for the presence of the n-grams
        removeContractions: (bool) if true, preprocesses input text to remove contractios (i.e. apostrophes between text)
        removePunctuation: (bool) if true, preprocesses input text to remove punctuation
        removeStopWords: (bool)if true, prepocesses input text to remove stop words, as per NTLK
        lowerCase: (bool) if true, preprocesses input to place all characters to lower case

    Output: 
        A function. See the output description for the function charFeatures below
    """    
    def charFeatures(text):
        """
        Input: 
            text: (str) the string to process 
        Output: 
            dictionary of extracted character n-grams (key) and their counts/indicator (value), removing whitespace 
        Example (n=3): "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, . . .}
        """
        #Remove CENSORED_TOKEN
        text = text.replace(CENSORED_TOKEN,'')

        #Remove Contractions
        if removeContractions == True:
            regex_pat = re.compile("([a-z]+'[a-z]+)", flags=re.IGNORECASE)
            text = re.sub(pattern=regex_pat,repl='',string=text)
    
        #Remove Punctuation
        if removePunctuation == True:
            for i in PUNCTUATION_LIST:
                text = text.replace(i, '')

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
        return result
    
    n = max(1,int(n))
    return charFeatures

#Test Cases
if False:
    extractor1 = extractCharFeatures(3)
    extractor2 = extractCharFeatures(4,removeStopWords=True)
    extractor3 = extractCharFeatures(4,removeStopWords=True,lowerCase=True)
    print extractor1("I like tacos")
    print extractor1("Land and air")
    print extractor2("Land and air")
    print extractor3("Land and air")
    extractor4 = extractCharFeatures(2)
    extractor5 = extractCharFeatures(2,removePunctuation=True)
    extractor6 = extractCharFeatures(2,removePunctuation=True, count=False)
    extractor7 = extractCharFeatures(2,removePunctuation=True,removeContractions=True)
    print extractor4("I ain't going, don't go!")
    print extractor5("I ain't going, don't go!")
    print extractor6("I ain't going, don't go!")
    print extractor7("I ain't going, don't go!")



##########################################################################################################
def extractWordFeatures(n, 
                count = True, removeContractions = False, removePunctuation = True, 
                removeStopWords = True, lowerCase = True):
    """
    Returns a function that takes as input a string |text| and returns a sparse vector (dictionary) 
    of n-gram word features.
    Whitespace is removed before forming n-grams

    Input: 
        n: (int) for n-grams (n>=1)

    Optional Args: 
        count: (bool) if true, returns the count of the n-grams
                      if false, returns an indicator for the presence of the n-grams
        removeContractions: (bool) if true, preprocesses input text to remove contractions (i.e. apostrophes between text)
        removePunctuation: (bool) if true, preprocesses input text to remove punctuation
        removeStopWords: (bool)if true, prepocesses input text to remove stop words, as per NTLK
        lowerCase: (bool) if true, preprocesses input to place all characters to lower case

    Output: 
        A function. See the output description for the function wordFeatures below
    """    
    def wordFeatures(text):
        """
        Input: 
            text: (str) the string to process 
        Output: 
            dictionary of extracted word n-grams (key) and their counts/indicator (value), removing whitespace 
        Example (n=2): "I like tacos" --> {'I like': 1, 'like tacos': 1}
        """
        #Remove CENSORED_TOKEN
        text = text.replace(CENSORED_TOKEN,'')

        #Remove Contractions
        if removeContractions == True:
            regex_pat = re.compile("([a-z]+'[a-z]+)", flags=re.IGNORECASE)
            text = re.sub(pattern=regex_pat,repl='',string=text)
    
        #Remove Punctuation
        if removePunctuation == True:
            for i in PUNCTUATION_LIST:
                text = text.replace(i, '')

        #Remove Capitalization
        if lowerCase == True:
            text = text.lower()

        #Split into a word list
        words = word_tokenize(text) #from nltk

        #Remove Stop Words
        if removeStopWords == True:
            words = [w for w in words if w not in STOP_WORDS]
    
        #Get Word Counts
        word_ngrams = ngrams(words,n)
        ngram_counts = collections.defaultdict(int)
        if count == True:
            for w in word_ngrams:
                ngram_counts[w] +=1
        else:
            for w in word_ngrams:
                ngram_counts[w] =1
        
        #Return Value
        return ngram_counts

    n = max(1,int(n))
    return wordFeatures


#Test Cases
if False:
    extractor1 = extractWordFeatures(1)
    extractor2 = extractWordFeatures(1,removeStopWords=True)
    extractor3 = extractWordFeatures(1,removeStopWords=True,lowerCase=True)
    print extractor1("I like tacos")
    print extractor1("Land and air")
    print extractor2("Land and air")
    print extractor3("Land and air")
    extractor4 = extractWordFeatures(2)
    extractor5 = extractWordFeatures(2,removePunctuation=True)
    extractor6 = extractWordFeatures(2,removePunctuation=True, count=False)
    extractor7 = extractWordFeatures(2,removePunctuation=True,removeContractions=True)
    print extractor4("I ain't going, don't go!")
    print extractor5("I ain't going, don't go!")
    print extractor6("I ain't going, don't go!")
    print extractor7("I ain't going, don't go!")
    print extractor4("I ate some cheese and ate some steak")
    print extractor5("I ate some cheese and ate some steak")
    print extractor6("I ate some cheese and ate some steak")
    print extractor7("I ate some cheese and ate some steak")



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
                features[(extractorName,key)] = value
        return features
    return extract

#Test Cases
if False:
    extractor1 = extractCharFeatures(3)
    extractor2 = featureExtractor([extractCharFeatures(3)])
    extractor3 = extractCharFeatures(4)
    extractor4 = featureExtractor([extractCharFeatures(3),extractCharFeatures(4)])
    extractor5 = featureExtractor([extractCharFeatures(3),extractWordFeatures(2)])
    print extractor1("I go shopping")
    print extractor2("I go shopping") #duplicate extractor1
    print extractor3("I go shopping")
    print extractor4("I go shopping") #combines extractor1 and extractor3
    print extractor5("I go shopping")


##########################################################################################################
def DesignMatrix(data,featurizer,num_train):
    """
    Takes as input a list (or 1-D array) of model input values x^(i) to which to apply a featurizer, phi()
    Produces train/dev sparse matrices which can be used for modelling    

    @param data: 1-dimensional array (list) of inputs
    @param featurizer: a function that generates a dictionary given an item from data
    @param num_train: number of examples to return in X_train. The rest go in X_dev
    """
    #for dev
    #data = data['description'].as_matrix()
    #featurizer = extractCharFeatures(3)
    #num_train = 600
    
    #----------------------------------------------------------------#
    # Experimenting with parallel processing - not working yet
    
    # This works if featurizer is a top-level function
    #feature_list = list(Pool().map(featurizer,data))

    # This hack doesn't work
    #def featurizer_wrapper(x):
    #    return featurizer(x)    
    #feature_list = list(Pool().map(featurizer_wrapper,data))
    #----------------------------------------------------------------#

    feature_list = list(map(featurizer,data))
    assert num_train > 0 and num_train < len(feature_list)

    vectorizer = DictVectorizer() 
    train_feature_mat_dense = vectorizer.fit_transform(feature_list[:num_train]).toarray()
    dev_feature_mat_dense   = vectorizer.transform(feature_list[num_train:]).toarray()
    #doc: -http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html-
        #transform(): Named features not encountered during fit or fit_transform will be silently ignored.

    train_feature_mat_sparse = sp.csr_matrix(train_feature_mat_dense)
    dev_feature_mat_sparse   = sp.csr_matrix(dev_feature_mat_dense)

    feature_names = list(map(str,vectorizer.get_feature_names()))
    
    return train_feature_mat_sparse,dev_feature_mat_sparse,feature_names


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