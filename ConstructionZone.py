# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 2018

@author: Luke Hanks and Trevor La Pay
"""

import argparse
import pandas as pd
import numpy as np
import math, operator, functools
import scipy.stats as stats


# MAP constants
classColumn = 61189
vocabularyLength = 61188
beta = 1/vocabularyLength
alpha = 1 + beta
allClasses = pd.read_csv("newsgrouplabels.txt", header=None)

classWordCounts = dict([(1, 154382), (2, 138499), (3, 116141), (4, 103535), (5, 90456),
                        (6, 144656), (7, 64094), (8, 107399), (9, 110928), (10, 124537), (11, 143087),
                        (12, 191242), (13, 97924), (14, 158930), (15, 162521), (16, 236747), (17, 172257),
                        (18, 280067), (19, 172670), (20, 135764)])
# Generate a dataframe for priors.
def generateMLEpriors(trainingDf):
    # totalClasses = len(allClasses.index)
    numDocs = len(trainingDf)
    priors = []
    for num, val in allClasses.iterrows():
        prior = len(trainingDf.loc[trainingDf[classColumn] == (num+1)])/numDocs
        priors.append(prior)
    return pd.DataFrame(priors)

# Approach: read in priors and map probability files.
# take the probability MAP matrix and apply log2 to all elements
# first, since we need to use those for log likelihoods.
# To classify a given row in the testing data, iterate through
# that file and do the following:
#   Calculate the likelihood that the row belongs to each class.
#   this is accomplished by multiplying the training data row (which includes word frequencies)
#   and the probability matrix row for the class in question.
#   Sum all values in this row, and add to the log 2 of the prior for that class.
#   Finally, add the sums to a list and pick the argmax out of that list, which
#   should be the likeliest class for the element in the data.
def naiveBayesClassify(trainingDf):
    # take argmax of the probability of a test example belonging to a given class across all features.
    priors = pd.read_csv("priors.csv", header=None).drop(0, axis=1).drop(0, axis=0)
    probabilityMatrix = pd.read_csv("map_probabilities.csv", header=None).drop(0, axis=1).drop(0, axis=0)
    # apply log2 first for calculating log sums
    probabilityMatrix = probabilityMatrix.applymap(math.log2)
    calculated_probabilities = []
    # classes = rows in priors vector.
    # wondering how to do this with matrix multiplication. Too slow iteratively.
    for row in trainingDf:
        for num, val in priors.iterrows():
            sumProb = trainingDf.iloc[row].multiply(probabilityMatrix.iloc[num - 1]).values.sum()
            calculated_probabilities.append(sumProb + math.log2(val.iloc[0]))
        print(np.argmax(calculated_probabilities))
        calculated_probabilities = []



# Generate a MAP dataframe for probabilites across classes and features.
def generateMAPmatrix(trainingDf):
    vocabDf = pd.read_csv("vocabulary.txt", header=None)
    listoflists = []
    for numClass, newsGroup in allClasses.iterrows():
        rowdata = []
        totalWordsInClass = classWordCounts.get(numClass + 1)
        trainingDfByClass = trainingDf.loc[trainingDf[61189] == (numClass + 1)]
        for numWord, word  in vocabDf.iterrows():
            if len(trainingDfByClass) > 1:
                # Counting the words in a class is too slow using dataframes.
                # There is probably a faster way to do this (either that, or create a
                # file that just stores these values in a vector or something)
                countWordsinClass = trainingDfByClass[numWord].sum()
            else:
                # don't skip zero counts since we are using MAP hallucinated values.
                countWordsinClass = totalWordsInClass = 0
            numerator = countWordsinClass + alpha - 1
            denominator = totalWordsInClass + ((alpha - 1)*vocabularyLength)
            rowdata.append(numerator/denominator)
        listoflists.append(rowdata)
        print(rowdata)
    return pd.DataFrame(listoflists)


def main():
    # generateSparseFiles() - we should discuss how to do this. We may not want to use pandas...
    # vocabFile = open('vocabulary.txt', "r")
    # allWords = vocabFile.read().splitlines()
    # colNames = ['docId'] + list(range(1, len(allWords) + 1)) + ['labelId']
    trainingDf = pd.read_csv("training.csv", header=None, nrows =300).to_sparse(fill_value=0)\
        .drop(0, axis=1).drop(61189, axis=1)
    # priorsdf = generateMLEpriors(trainingdf)
    # priorsdf.to_csv("priors.csv")
    # mapdf = generateMAPmatrix(trainingdf)
    # mapdf.to_csv("map_probabilities.csv")
    # I'm running this against a tiny subset of the training to see if the classifier is working.
    # Currently, it is not. 
    naiveBayesClassify(trainingDf)

if __name__ == "__main__": main()
