import pandas as pd
import collections
from collections import defaultdict
import re
import nltk
from nltk.util import ngrams
import matplotlib.pyplot as plt
import operator

def importData(path):
    data = pd.DataFrame.from_csv(path)
    
    return data
    
def exportData(data, path):
    data.to_csv(path)

def main() :
    df = importData('./winemag-data_first150k.csv')

    grapeVarietyDict = defaultdict(list)
    grapeVarietyByCountryDict = {}
    for index, row in df.iterrows(): 
        grapeVariety = row['variety']
        grapeCountry = row['country']

        #if grapeVariety not in grapeVarietyByCountryDict :
        if grapeCountry not in grapeVarietyByCountryDict :
            #grapeVarietyByCountryDict[grapeVariety] = defaultdict(float)
            grapeVarietyByCountryDict[grapeCountry] = defaultdict(float)
        if grapeVariety not in grapeVarietyDict :
            grapeVarietyDict[grapeVariety].append(0)
        #grapeVarietyByCountryDict[grapeVariety][grapeCountry] += 1
        grapeVarietyByCountryDict[grapeCountry][grapeVariety] += 1
        grapeVarietyDict[grapeVariety][0] += 1

    #print grapeVarietyByCountryDict
    for key, value in grapeVarietyDict.items() :
        print key, value
        if value[0] < 1000 :
            del grapeVarietyDict[key]

    '''
    for key, value in grapeVarietyByCountryDict.items() :
        value = sum(grapeVarietyByCountryDict[key].values())
        print "Key {}, value {}".format(key, sum(grapeVarietyByCountryDict[key].values()))
        if value < 100 :
            del grapeVarietyByCountryDict[key]

    '''
    #sorted_d = sorted(grapeVarietyByCountryDict.items(), key=operator.itemgetter(1),reverse=True)
    #print('Dictionary in descending order by value : ',sorted_d)
    df2 = pd.DataFrame(grapeVarietyDict)
    print "DF2: ", df2
    print "Columns:", df2.columns
    columnsList = list(df2.columns)
    valuesList = []
    for column in columnsList:
        valuesList.append(df2[column].values[0])

    df = pd.DataFrame(dict(A = columnsList, B = valuesList))

    print "DF: ", df
    #df3 = pd.DataFrame(grapeVarietyByCountryDict)


    #df3.plot(kind = 'bar', stacked=True, figsize=(100,20))
    #print "DF3:", df3

    df.set_index('A').plot.bar(rot = 90, figsize=(30, 20))

    plt.savefig('ps1q1c.png')
    #print grapeVarietyDict

    #print grapeVarietyByCountryDict

if __name__ == '__main__':
    main()
'''
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
'''