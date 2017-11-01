#test.py
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
    #print "Performing tfidf on: ", word, blob, bloblist
    return tf(word, dicta) * idf(word, dictList)

#Fancy
def makeDictList(document, refDocumentList) :
    bloblist = [tb(document)]
    for document in refDocumentList :
        bloblist.append(tb(document))
    return bloblist

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

def extractTDIFWordFeatures(document, refDocumentList, k = None) :
    bloblist = makeDictList(document, refDocumentList)
    docLength = len(document.split())
    if k == None or k > docLength or k < 1 :
        k = docLength 
    print("Top words in document: ")
    blob = bloblist[0] #Document
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:k]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, k)))
        
def extractTDIFWordFeatureAll(documentList, keyList, k) :
    dictList = makeDictListAll(documentList, keyList)
    for i, dicta in enumerate(dictList):
        print("Top words in document {} which has key {}".format(i + 1, keyList[i]))
        scores = {word: tfidf(word, dicta, dictList) for word in dicta.keys()}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:k]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, k)))
   
document1 = """welcome to stay in private room with queen bed and detached private bathroom on the second floor."""

document2 = """stylish, fully remodeled home in upscale NW Alamo Heights Area."""

document3 = """private room (shared bathroom) in a warm family home. it's perfect for business trips, vacation and extended stays."""

def main():
    documentList = [document1, document2, document3]
    extractTDIFWordFeatureAll(documentList, 3)

if __name__ == '__main__':
    main()