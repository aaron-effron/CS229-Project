#tfidf.py.  TF-IDF analysis of wine descriptions.
#Base code borrowed from http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
from __future__ import division, unicode_literals
import math
from textblob import TextBlob as tb
from collections import defaultdict

def tf(word, dicta):
    return float(dicta[word]) / sum(dicta.values())

def n_containing(word, dictList):
    return sum(1 for dicta in dictList if dicta[word] > 0)

def idf(word, dictList):
    return math.log(len(dictList) / (1 + n_containing(word, dictList)))

def tfidf(word, dicta, dictList):
    return tf(word, dicta) * idf(word, dictList)

#Use dictionaries for tfidf
def makeDictListAll(documentList, keyList) :
    dictList = []
    for i in range(0, len(documentList)) :
        document = documentList[i]
        documentDict = defaultdict(float)
        for word in document.split() :
            add = True
            for key in keyList :
                if key in word.lower() :
                    add = False
                    break
            if (add) :
                documentDict[word.lower()] += 1

        dictList.append(documentDict)
    return dictList
        
def extractTFIDFWordFeatureAll(documentList, keyList, k) :
    dictList = makeDictListAll(documentList, keyList)
    for i, dicta in enumerate(dictList):
        print("Top words in document {} which has key {}".format(i + 1, keyList[i]))
        scores = {word: tfidf(word, dicta, dictList) for word in dicta.keys()}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:k]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, k)))
   