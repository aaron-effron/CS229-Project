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
import tfidf_rev
from matplotlib import rcParams

def importData(path):
    data = pd.DataFrame.from_csv(path)
    try :
        data = pd.read_csv(path, encoding = 'utf8')
    except UnicodeDecodeError as e :
        print "UGH!", e
    
    return data
    
def exportData(data, path):
    data.to_csv(path)

def getDataFrame() :
    df = importData('./winemag-data_first150k_2.csv')
    return df

def getGrapeVarietyDict(df, excludeSmall = False) :
    grapeVarietyDict = defaultdict(list)
    for index, row in df.iterrows(): 
        grapeVariety = row['variety']
        if 'Blend' in grapeVariety.split() :
            continue
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
    print "HI"
    for index, row in df.iterrows(): 
        grapeVariety = row['variety']
        price = row['price']
        if math.isnan(price) :
            continue
        if 'Blend' in grapeVariety.split() :
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

    return grapeVarietyPriceDict, standardDevDict

def getDescriptionDict(df, excludeSmall = False) :
    descriptionDict = defaultdict(list)
    for index, row in df.iterrows(): 
        grapeVariety = row['variety']
        description = row['description']

        puctuationList = ['.',',','?','!',"'",'"',':',';','-'] #doesn't remove unicode punctuation
        for i in puctuationList:
            description = description.replace(i, '')
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

    return descriptionDict

def plotBarChart(grapeVarietyDict) :
    rcParams.update({'figure.autolayout': True})
    df2 = pd.DataFrame(grapeVarietyDict)
    columnsList = list(df2.columns)
    valuesList = []
    for column in columnsList:
        valuesList.append(df2[column].values[0])

    df = pd.DataFrame(dict(grapeVariety = columnsList, Count = valuesList))
    df = df.sort_values(['Count'], ascending=[0]).reset_index(drop=True)

    df.set_index('grapeVariety').plot.bar(rot = 90)
    plt.savefig('BarChart.png')

def plotStackedBarChart(grapeVarietyByCountryDict) :
    rcParams.update({'figure.autolayout': True})
    df3 = pd.DataFrame(grapeVarietyByCountryDict)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, 14))
    df3.plot(kind = 'bar', stacked=True, color=colors).legend(loc='upper right', fontsize=5)#, bbox_to_anchor=(1, 0.5))
    plt.savefig('stackedBarChart.png')

def plotPriceBarChart(grapeVarietyPriceDict, standardDevDict) :
    rcParams.update({'figure.autolayout': True})
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
        stdDevValuesList.append(df3[df.values[i][0]].values[0])
        stdDevColumnsList.append(df.values[i][0])

    errors = pd.DataFrame(dict(B = stdDevValuesList))

    _, ax= plt.subplots() 
    prices = df[['B']]

    prices.plot(kind='bar', yerr=errors, ax=ax, error_kw=dict(ecolor='black',elinewidth=0.5), grid=False, rot = 90)
    ax.set_xticklabels(df.A)

    #df.set_index('A').plot(kind='bar', yerr=errors, error_kw=dict(ecolor='black',elinewidth=0.5), grid=False, rot = 90, figsize=(30, 20))
    plt.savefig('PriceBarChart.png')

def main() :
    reload(sys)  
    sys.setdefaultencoding('utf8')
    df = getDataFrame()
    for arg in sys.argv :
        if arg == 'v' :
            grapeVarietyDict = getGrapeVarietyDict(df, True)
            grapeVarietyByCountryDict = getGrapeVarietyByCountryDict(df, grapeVarietyDict, True)
            #plotBarChart(grapeVarietyDict)
            plotStackedBarChart(grapeVarietyByCountryDict)
        elif arg == 'p' :
            grapeVarietyPriceDict, standardDevDict = getGrapeVarietyPriceDict(df, excludeSmall=True)
            plotPriceBarChart(grapeVarietyPriceDict, standardDevDict)  
    
        elif arg == 'd' :
            descriptionDict = getDescriptionDict(df, excludeSmall=True)
            keyList = []
            documentList = []
            for key in descriptionDict :
                document1 = descriptionDict[key]
                if "Blend" in key.split() : # Don't include blends
                    continue
                for keya in key.split() : #break up multiple word names
                    keyList.append(keya.lower())
                documentList.append(descriptionDict[key][0])
            tfidf_rev.extractTDIFWordFeatureAll(documentList, keyList, 3)
           
if __name__ == '__main__':
    main()