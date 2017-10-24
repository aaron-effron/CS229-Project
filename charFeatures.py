import collections
import re
from nltk.corpus import stopwords as nltk_stopwords

PUNCTUATION_LIST = ['.',',','?','!',"'",'"',':',';','-']
STOP_WORDS = nltk_stopwords.words('english')

def extractCharFeatures(n,
        count=True,removeContractions=False,removePunctuation=False,
        removeStopWords=False, lowerCase=False):
    """
    Returns a function that takes an input a string |text| and returns a sparse vector (dictionary) 
    of n-gram features.
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
        #Remove Contractions
        if removeContractions == True:
            contractions = re.findall("([a-zA-Z]+'[a-zA-Z]+)", text)
            for word in contractions:
                text = text.replace(word, '')
    
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