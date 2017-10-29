import pandas as pd
import collections
import re
import nltk
from nltk.util import ngrams

def importData(path):
    data = pd.DataFrame.from_csv(path)
    
    return data
    
def exportData(data, path):
    data.to_csv(path)

def extractWordFeatures(description, n, contractions = True, puctuation = True, cases = True, stopWords = True, count = True):
    #fix hexcode
    m = re.findall (r'[\x80-\xff][\x80-\xff][\x80-\xff]', description)
    for word in m:
        description = description.decode('utf-8')
    
    #contractions
    if contractions == True:
        m = re.findall("([a-zA-Z]+'[a-zA-Z]+)", description)
        for word in m:
            description = description.replace(word, '')
    
    #punctuation
    if puctuation == True:
        puctuationList = ['.',',','?','!',"'",'"',':',';','-'] #doesn't remove unicode punctuation
        for i in puctuationList:
            description = description.replace(i, '')

            #capitalization
    if cases == True:
        description = description.lower()
    
    #tokenize (note: n values less than one default to one and float values are rounded)                              
    description = nltk.word_tokenize(description)
    
    #remove stop words
    if stopWords == True:
        description = [word for word in description if word not in nltk.corpus.stopwords.words('english')]
    
    grams = ngrams(description,n)
    if count == True:
        return collections.Counter(grams)
    else:
        return collections.Counter(grams).elements()