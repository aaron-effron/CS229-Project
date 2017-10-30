import pandas as pd
import collections
from collections import defaultdict
import re
import nltk
from nltk.util import ngrams
import matplotlib.pyplot as plt
import operator
import numpy as np
import math
import sys  
import tfidf



def importData(path):
    data = pd.DataFrame.from_csv(path)
    
    return data
    
def exportData(data, path):
    data.to_csv(path)

def getDataFrame() :
    df = importData('./winemag-data_first150k_toy.csv')
    return df

def getGrapeVarietyDict(df, excludeSmall = False) :
    grapeVarietyDict = defaultdict(list)
    for index, row in df.iterrows(): 
        grapeVariety = row['variety']
        if grapeVariety not in grapeVarietyDict :
            grapeVarietyDict[grapeVariety].append(0)
        grapeVarietyDict[grapeVariety][0] += 1

    #Prune the dictionary, so the resulting chart isn't crazy
    if (excludeSmall) :
        for key, value in grapeVarietyDict.items() :
            if value[0] < 1000 :
                del grapeVarietyDict[key]
    
    return grapeVarietyDict

def getGrapeVarietyByCountryDict(df, grapeVarietyDict, excludeSmall = False) :
    grapeVarietyByCountryDict = {}
    for index, row in df.iterrows(): 
        grapeCountry = row['country']
        grapeVariety = row['variety']

        if grapeVariety not in grapeVarietyDict.keys() : #Only including common ones
            continue

        if grapeCountry not in grapeVarietyByCountryDict :
            grapeVarietyByCountryDict[grapeCountry] = defaultdict(float)
        grapeVarietyByCountryDict[grapeCountry][grapeVariety] += 1

    #Delete countries that produce less than 200 wine varities
    if (excludeSmall) :
        for key, value in grapeVarietyByCountryDict.items() :
            value = sum(grapeVarietyByCountryDict[key].values())
            if value < 200 :
                del grapeVarietyByCountryDict[key]
    
    return grapeVarietyByCountryDict


def getGrapeVarietyPriceDict(df, excludeSmall = False) :
    grapeVarietyPriceDict = defaultdict(list)
    for index, row in df.iterrows(): 
        print row['variety']
        grapeVariety = row['variety'].decode('utf-8')
        price = row['price']
        if math.isnan(price) :
            continue
        grapeVarietyPriceDict[grapeVariety].append(price)

    #Prune the dictionary, so the resulting chart isn't crazy
    if excludeSmall :
        for key, value in grapeVarietyPriceDict.items() :
            if len(value) < 1000 :
                del grapeVarietyPriceDict[key]
    
    #Now set each index to be the average, and set standard deviations
    standardDevDict = defaultdict(list)
    for key, value in grapeVarietyPriceDict.items() :
        newValue = []
        standardDevDict[key].append(np.std(value))
        newValue.append(sum(grapeVarietyPriceDict[key])/len(grapeVarietyPriceDict[key]))
        grapeVarietyPriceDict[key] = newValue

    return grapeVarietyPriceDict, 

def getDescriptionDict(df, excludeSmall = False) :
    descriptionDict = defaultdict(list)
    for index, row in df.iterrows(): 
        grapeVariety = row['variety']
        print row['description']
        description = row['description'].decode('utf-8')
        descriptionDict[grapeVariety].append(description)

    #Prune the dictionary, so the resulting chart isn't crazy
    if excludeSmall :
        for key, value in descriptionDict.items() :
            if len(value) < 1000 :
                del descriptionDict[key]
    
    #Now set each value to be all reviews appended
    for key in descriptionDict :
        newValue = [' '.join(descriptionDict[key])]
        descriptionDict[key] = newValue

    print descriptionDict
    return descriptionDict

def plotBarChart(grapeVarietyDict) :
    df2 = pd.DataFrame(grapeVarietyDict)
    columnsList = list(df2.columns)
    print columnsList
    valuesList = []
    for column in columnsList:
        valuesList.append(df2[column].values[0])

    df = pd.DataFrame(dict(A = columnsList, B = valuesList))
    df.set_index('A').plot.bar(rot = 90, figsize=(30, 20))
    plt.savefig('BarChart.png')

def plotStackedBarChart(grapeVarietyByCountryDict) :
    df3 = pd.DataFrame(grapeVarietyByCountryDict)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, 14))
    df3.plot(kind = 'bar', stacked=True, figsize=(30,20), color=colors)
    plt.savefig('stackedBarChart')

def plotPriceBarChart(grapeVarietyPriceDict, standardDevDict) :
    df2 = pd.DataFrame(grapeVarietyPriceDict)
    #print df2
    columnsList = list(df2.columns)
    valuesList = []
    for column in columnsList:
        valuesList.append(df2[column].values[0])

    df = pd.DataFrame(dict(A = columnsList, B = valuesList))
    df = df.sort_values(['B'], ascending=[0]).reset_index(drop=True)

    #Can parse through df to make sure the ordering is correct

    df3 = pd.DataFrame(standardDevDict)
    stdDevColumnsList = []
    stdDevValuesList = []

    for i in range(0, len(df.values)) :
        print df.values[i]
        stdDevValuesList.append(df3[df.values[i][0]].values[0])
        stdDevColumnsList.append(df.values[i][0])

    print stdDevValuesList

    #errors = pd.DataFrame(dict(A = stdDevColumnsList, B = stdDevValuesList))
    errors = pd.DataFrame(dict(B = stdDevValuesList))
    print errors
    print df

    _, ax= plt.subplots() 
    prices = df[['B']]
    print prices

    prices.plot(kind='bar', yerr=errors, ax=ax, error_kw=dict(ecolor='black',elinewidth=0.5), grid=False, rot = 90, figsize=(30, 20))
    ax.set_xticklabels(df.A)
    print df.A

    #df.set_index('A').plot(kind='bar', yerr=errors, error_kw=dict(ecolor='black',elinewidth=0.5), grid=False, rot = 90, figsize=(30, 20))
    plt.savefig('PriceBarChart.png')


def main() :
    reload(sys)  
    sys.setdefaultencoding('utf8')
    df = getDataFrame()
    #grapeVarietyDict = getGrapeVarietyDict(df)
    #grapeVarietyByCountryDict = getGrapeVarietyByCountryDict(df, grapeVarietyDict)
    #grapeVarietyPriceDict, standardDevDict = getGrapeVarietyPriceDict(df)
    descriptionDict = getDescriptionDict(df)
    for key in descriptionDict :
        document1 = descriptionDict[key]
        documentList = [descriptionDict[keya][0] for keya in descriptionDict if key != keya]
        tfidf.extractTDIFWordFeatures(document1[0], documentList, -1)
    #print grapeVarietyDict
    #print grapeVarietyPriceDict
    #plotPriceBarChart(grapeVarietyPriceDict, standardDevDict)
   

    #print grapeVarietyPriceDict
    #plotBarChart(grapeVarietyDict)
    #plotStackedBarChart(grapeVarietyByCountryDict)
    
    
if __name__ == '__main__':
    main()