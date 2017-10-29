import pandas as pd
import collections
from collections import defaultdict
import re
import nltk
from nltk.util import ngrams
import matplotlib.pyplot as plt
import operator
import numpy as np

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

        #if grapeVariety not in grapeVarietyByCountryDict :
        if grapeVariety not in grapeVarietyDict :
            grapeVarietyDict[grapeVariety].append(0)
        grapeVarietyDict[grapeVariety][0] += 1

     #print grapeVarietyByCountryDict
    for key, value in grapeVarietyDict.items() :
        print key, value
        if value[0] < 1000 :
            del grapeVarietyDict[key]

    #Parse through again, don't include non-common grape varieties
    #COMMENT OUT WHEN RUNNING BAR CHART
    for index, row in df.iterrows(): 
        grapeCountry = row['country']
        grapeVariety = row['variety']

        if grapeVariety not in grapeVarietyDict.keys() : #Only including common ones
            continue

        #if grapeVariety not in grapeVarietyByCountryDict :
        if grapeCountry not in grapeVarietyByCountryDict :
            grapeVarietyByCountryDict[grapeCountry] = defaultdict(float)
        #grapeVarietyByCountryDict[grapeVariety][grapeCountry] += 1
        grapeVarietyByCountryDict[grapeCountry][grapeVariety] += 1

   

    #Delete countries that produce less than 200 wine varities
    for key, value in grapeVarietyByCountryDict.items() :
        value = sum(grapeVarietyByCountryDict[key].values())
        print "Key {}, value {}".format(key, sum(grapeVarietyByCountryDict[key].values()))
        if value < 200 :
            del grapeVarietyByCountryDict[key]

    
    #sorted_d = sorted(grapeVarietyByCountryDict.items(), key=operator.itemgetter(1),reverse=True)
    #print('Dictionary in descending order by value : ',sorted_d)
    '''
    df2 = pd.DataFrame(grapeVarietyDict)
    print "DF2: ", df2
    print "Columns:", df2.columns
    columnsList = list(df2.columns)
    valuesList = []
    for column in columnsList:
        valuesList.append(df2[column].values[0])

    df = pd.DataFrame(dict(A = columnsList, B = valuesList))

    print "DF: ", df
    '''
    df3 = pd.DataFrame(grapeVarietyByCountryDict)

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, 14))
    df3.plot(kind = 'bar', stacked=True, figsize=(30,20), color=colors)
    #print "DF3:", df3

    #df.set_index('A').plot.bar(rot = 90, figsize=(30, 20))

    #plt.savefig('ps1q1c.png')
    plt.savefig('stackedBarChart')


if __name__ == '__main__':
    main()