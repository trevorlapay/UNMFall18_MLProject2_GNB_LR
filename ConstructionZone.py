# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 2018

@author: Luke Hanks and Trevor La Pay
"""

import argparse
import pandas as pd
import math, operator, functools
import scipy.stats as stats


# MAP constants
classColumn = 61189
vocabularyLength = 61188
beta = 1/vocabularyLength
alpha = 1 + beta
allClasses = pd.read_csv("newsgrouplabels.txt", header=None)

# Generate a dataframe for priors.
def generateMLEpriors(trainingDf):
    # totalClasses = len(allClasses.index)
    numDocs = len(trainingDf)
    priors = []
    for num, val in allClasses.iterrows():
        prior = len(trainingDf.loc[trainingDf[classColumn] == num])/numDocs
        priors.append(prior)
    return pd.DataFrame(priors)

#Generate a MAP dataframe for probabilites across classes and features.
def generateMAPmatrix(trainingDf):
    vocabDf = pd.read_csv("vocabulary.txt", header=None)
    listoflists = []
    for numClass, newsGroup in allClasses.iterrows():
        rowdata = []
        for numWord, word in vocabDf.iterrows():
            trainingDfByClass = trainingDf.loc[trainingDf[61189] == numClass]
            if len(trainingDfByClass) > 1:
                # Counting the words in a class is too slow using dataframes.
                # There is probably a faster way to do this (either that, or create a
                # file that just stores these values in a vector or something)
                countWordsinClass = trainingDfByClass.iloc[numWord].sum()
                totalWordsInClass = getTotalWords(trainingDfByClass)
            else:
                # don't skip zero counts since we are using MAP hallucinated values.
                countWordsinClass = totalWordsInClass = 0
            numerator = countWordsinClass + alpha - 1
            denominator = totalWordsInClass + ((alpha - 1)*vocabularyLength)
            rowdata.append(numerator/denominator)
        listoflists.append(rowdata)
    return pd.DataFrame(listoflists)

def getTotalWords(trainingDfByClass):
    totalWordCount = 0
    # I need to get the sum of all the words for a given class here.
    # this is absolutely broken (it returns the same value every time). I'm leaving it as a placeholder for now.
    totalWords = trainingDfByClass.iloc[0:61188].sum(axis=1)
    for count in totalWords:
        totalWordCount = totalWordCount + count
    return totalWordCount










def generateSparseFiles(trainingDataFile="training.csv"):
    trainingdf = pd.read_csv(trainingDataFile, header=None, nrows=100)
    sparse = trainingdf.to_sparse()
    # this is not sparse - it is loaded with zeros. For now just restricting the number of rows.
    # for now, only generate training (until we get the rest fleshed out)
    sparse.to_csv("sparse_training_tiny.csv")



def main():
    # generateSparseFiles() - we should discuss how to do this. We may not want to use pandas...
    trainingdf = pd.read_csv("sparse_training_tiny.csv", header=None)
    priorsdf = generateMLEpriors(trainingdf)
    mapdf = generateMAPmatrix(trainingdf)

if __name__ == "__main__": main()
