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
def makeBlobList(documentList) :
    bloblist = []
    for document in documentList :
        bloblist.append(tb(document))
    return bloblist



document1 = """Welcome to stay in private room with queen bed and detached private bathroom on the second floor. 
	Another private bedroom with sofa bed is available for additional guests. 10$ for an additional guest.\n
	10min from IAH airport\nAirport pick-up/drop off is available for $10/trip.
	"""

document2 = """Stylish, fully remodeled home in upscale NW Alamo Heights Area. 
	\n\nAmazing location - House conveniently located in quiet street, with beautiful seasoned trees, 
	prestigious neighborhood and very close to the airport, 281, 410 loop and down-town area. 
	 Driveway can park 4 cars.
	 """

#Cheap
document3 = """Welcome to stay in private room with queen bed and detached private bathroom on the second floor. 
	Another private bedroom with sofa bed is available for additional guests. 10$ for an additional guest.\n
	10min from IAH airport\nAirport pick-up/drop off is available for $10/trip.

Private Room (shared bathroom) in a warm family home. It's perfect for business trips, vacation and extended stays. 
If you are in town as a childbirth tourist, this is also a perfect place for you.
	"""

def main():
    documentList = [document1, document2, document3]
    bloblist = makeBlobList(documentList)
    for i, blob in enumerate(bloblist):
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:5]:
            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

if __name__ == '__main__':
    main()