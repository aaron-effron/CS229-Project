from charFeatures import *

def featureExtractor(individualExtractors):
    """
    Takes as input a list of feature extractor functions,
    each returning a sparse vector, and returns a function 
    that combines their output into a single sparse vector output.
    """
    def extract(x):
        features = collections.defaultdict(float)
        for extractor in individualExtractors:
            extractorName = extractor.func_name
            for key, value in extractor(x).items():
                #append name because there may be identical keys for different extractors
                features[(extractorName,key)] = value
        return features
    return extract

#Test Cases
if False:
    extractor1 = extractCharFeatures(3)
    extractor2 = featureExtractor([extractCharFeatures(3)])
    extractor3 = extractCharFeatures(4)
    extractor4 = featureExtractor([extractCharFeatures(3),extractCharFeatures(4)])
    print extractor1("I go shopping")
    print extractor2("I go shopping") #duplicate extractor1
    print extractor3("I go shopping")
    print extractor4("I go shopping") #combines extractor1 and extractor3