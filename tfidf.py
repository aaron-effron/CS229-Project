#test.py
from __future__ import division, unicode_literals
import math
from textblob import TextBlob as tb


def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

#Fancy
def makeBlobList(document, refDocumentList) :
    bloblist = [tb(document)]
    for document in refDocumentList :
        bloblist.append(tb(document))
    return bloblist

def makeBlobListAll(documentList) :
    bloblist = []
    for document in documentList :
        bloblist.append(tb(document))
    return bloblist

def extractTDIFWordFeatures(document, refDocumentList, k = None) :
    bloblist = makeBlobList(document, refDocumentList)
    docLength = len(document.split())
    if k == None or k > docLength or k < 1 :
        k = docLength 
    print("Top words in document: ")
    blob = bloblist[0] #Document
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:k]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, k)))
        
def extractTDIFWordFeatureAll(documentList, k) :
    bloblist = makeBlobListAll(documentList)
    for i, blob in enumerate(bloblist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
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