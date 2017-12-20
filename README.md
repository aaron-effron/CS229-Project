# CS229-Project: Learning the Language of Wine

Wine has been an integral element of human society for millennia. Accounts of wine date as far back as Egyptian and Roman times and wine often has symbolic significance for certain religions. In modern times, wine is often the beverage of choice for social gatherings and celebrations.

Unfortunately, despite its historical and contemporary significance, wine can often seem daunting and intimidating to many due to the oft use of strange and esoteric descriptors by wine professionals, or \textit{sommeliers} (from French: "wine steward"). Thus, we are motivated to ask the question: can machine learning help demystify wine and make it more accessible?

Our broad task is to "learn the language of wine." More concretely however, our aim is to build models that take as input the description of a wine by a wine expert and output:
- Grape type: red or white (binary classification)
- Grape variety (multi-class classification)
- Similar wines (recommendation)

Files:
- EDA.py: Exploratory data analysis, generating various plots
- decisionTree.py: Decision tree and random forest models
- grapeVarietyPlots.py: Also part of exploratory data analysis to generate various plots
- kMeansClustering.py: K-Means Clustering for wine recommendations
  (not included in report for space reasons and because of a better approach to recommendations)
- logistic.py: Logistic classification model
- multi_class_svm.py: Multi-class SVM (one-vs-n) model
- nb.py: Naive Bayes model
- processData.py: Preprocessing and input of data
- tfidf.py: Routines for computing TF-IDF significance
- util.py: Common utilities used throughout models
- w2v.py: Code to generate word2vec recommendations and plot the variety map

