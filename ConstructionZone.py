# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 2018

@author: Luke Hanks and Trevor La Pay
"""

import argparse
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn

# MAP constants
classColumn = 61189
vocabularyLength = 61188
# Instructions say to make beta = 1/vocabularyLength
beta = .01
alpha = 1 + beta
classesFile = open('newsgrouplabels.txt', "r")
allClasses = classesFile.read().splitlines()

# word counts won't change (unless we change the training set)
# The following 2 constants are ONLY to be used with the entire training set,
#  otherwise generate them from file..
classWordCounts = dict([(1, 154382), (2, 138499), (3, 116141), (4, 103535),
                        (5, 90456), (6, 144656), (7, 64094), (8, 107399), 
                        (9, 110928), (10, 124537), (11, 143087), (12, 191242),
                        (13, 97924), (14, 158930), (15, 162521), (16, 236747),
                        (17, 172257), (18, 280067), (19, 172670), (20, 135764)])

# Priors (these won't change unless we change the training set)
priors = [.04025, .052, .051833333333333335, .05358333333333333, 
          .050166666666666665, .0525, .0515, .051166666666666666,
          .05408333333333333, .052333333333333336, .05383333333333333, .05325,
          .05216666666666667, .05175, .05308333333333334, .05425, 
          .04833333333333333, .049416666666666664, .03891666666666667, 
          .035583333333333335]

# Accuracies at various beta levels for plotting. Reference: full training set.
beta_accuracies = [(.00001, .88367), (.000016, .88367), (.0001, .88957), 
                   (.001, .89371), (.01, .89784), (.1, .89193), (.2, .88662), 
                   (.5, .87835), (.8, .86625), (1, .85562)]


# Approach: read from priors list and map probability files.
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
def naiveBayesClassify(testingDf):
    # take argmax of the probability of a test example belonging to a given class across all features.
    # drop doc_id column and label column.
    testingDf = testingDf.drop('id', axis=1)
    probabilityMatrix = pd.read_csv("map_probabilities.csv", header=None).drop(0, axis=1).drop(0, axis=0)
    # apply log2 first for calculating log sums
    probabilityMatrix = probabilityMatrix.applymap(math.log2)
    calculated_probabilities = []
    classList = []
    # priors = pd.read_csv("priors.csv", header=None).iloc[:, 1].tolist()
    # classes = rows in priors vector.
    # wondering how to do this with matrix multiplication. Too slow iteratively.
    for index, row in testingDf.iterrows():
        for num, val in enumerate(priors):
            sumProb = row.multiply(probabilityMatrix.iloc[num]).values.sum()
            calculated_probabilities.append(sumProb + math.log2(val))
        classList.append((np.argmax(calculated_probabilities) + 1))
        calculated_probabilities = []
    labelCol = pd.Series(classList)
    return labelCol


# Generate a MAP dataframe for probabilites across classes and features.
# This needs to be updated every time the beta value changes.
def generateMAPmatrix(trainingDf):
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    # use below if using something other than the full training set.
    # classWordCounts = pd.read_csv("label_counts.csv", header=None).iloc[:, 1].tolist()
    listoflists = []
    for numClass, newsGroup in enumerate(allClasses):
        rowdata = []
        totalWordsInClass = classWordCounts.get(numClass + 1)
        trainingDfByClass = trainingDf.loc[trainingDf['labelId'] == (numClass + 1)]
        for numWord, word  in enumerate(allWords):
            if len(trainingDfByClass) > 1:
                countWordinClass = trainingDfByClass[numWord+1].sum()
            else:
                # don't skip zero counts since we are using MAP hallucinated values.
                countWordinClass = totalWordsInClass = 0
            numerator = countWordinClass + alpha - 1
            denominator = totalWordsInClass + ((alpha - 1)*vocabularyLength)
            rowdata.append(numerator/denominator)
        listoflists.append(rowdata)
    pd.DataFrame(listoflists).to_csv("map_probabilities.csv")

def generateSubmissionFileNB(testingDataFile="testing.csv", answersDataFile="answers.csv"):
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    colNames = ['id'] + list(range(1, len(allWords) + 1))
    testingDF = pd.read_csv(testingDataFile, header=None, names=colNames)
    answerDF = testingDF.filter(['id'], axis=1)
    answerDF['class'] = naiveBayesClassify(testingDF)
    answerDF.to_csv(answersDataFile, index=False)

def plotBetaValues():
    plt.semilogx(*zip(*beta_accuracies))
    plt.xlabel("Beta Value")
    plt.ylabel("Accuracy")
    plt.show()

#classify against a testing set and chart how well the algorithm classified.
def generateConfusionPlot(testingDf):
    labelCol = naiveBayesClassify(testingDf)
    testingDf['predictedClass'] = labelCol.values
    testingDf.drop(testingDf.columns.to_series()[1:61189], axis=1, inplace=True)
    listoflists = []
    for numClass, newsgroup1 in enumerate(allClasses):
        row = []
        for numPredictedClass, newsgroup2 in enumerate(allClasses):
            sum_match = len(testingDf.loc[(testingDf['labelId'] == (numClass + 1))
                                                      & (testingDf['predictedClass']
                                                         == (numPredictedClass + 1))])
            row.append(sum_match)
        listoflists.append(row)

    df_cm = pd.DataFrame(listoflists, range(20),
                             range(20))
    sn.heatmap(df_cm, annot=True)
    plt.show()


# split the training set into both a training and testing set at an index
# currently these are split roughly in half. Run this to generate
# large split files (these are not in the repo).
def splitTrainingTesting(splitAt):
    trainingDfStart = loadTraining()
    trainingDf = trainingDfStart[0:splitAt]
    testingDf = trainingDfStart[splitAt + 1:len(trainingDfStart)]
    trainingDf.to_csv("training_split.csv")
    testingDf.to_csv("testing_split.csv")
    return trainingDf, testingDf

# loads the training file from csv. Either whole or the split version.
def loadTraining(split=False):
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    colNames = ['id'] + list(range(1, len(allWords) + 1)) + ['labelId']
    if split:
        trainingFile = "training_split.csv"
    else:
        trainingFile = "training.csv"

    return pd.read_csv(trainingFile, header=None,
                             names=colNames).to_sparse(fill_value=0)

# loads the training files and testing. Assume split.
def loadTrainingAndTestingFromFile():
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    colNames = ['id'] + list(range(1, len(allWords) + 1)) + ['labelId']
    trainingDf = pd.read_csv("training_split.csv", header=None,
                             names=colNames).to_sparse(fill_value=0)
    testingDf = pd.read_csv("testing_split.csv", header=None,
                             names=colNames).to_sparse(fill_value=0)
    return trainingDf, testingDf

# generate priors to file
def generatePriors(trainingDf):
    labelProportions = trainingDf['labelId'].value_counts(normalize=True)
    labelProportions.to_csv("priors.csv")

# generate the class counts for MAP matrix dynamically (do this when the training set changes)
def generateClassCounts(trainingDf):
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    vocabFile.close()
    numWords = len(allWords)
    allWordIds = list(range(1, numWords + 1))
    labelWordSums = trainingDf.groupby('labelId')[allWordIds].agg(np.sum).to_sparse(fill_value=0)
    labelSums = labelWordSums.sum(axis=1)
    labelSums.to_csv("label_counts.csv")

# regenerate all static data about the training set.
# run this if you change the training set (i.e. run once after splitting the training set at a
# given row index)
def generateAll(trainingDf):
    # If you're using the full training set, don't bother loading these from files.
    # generateClassCounts(trainingDf)
    # generatePriors(trainingDf)
    generateMAPmatrix(trainingDf)
    # if you're loading the split files, create them with the below.
    # splitTrainingTesting(1200)

def main():
    generateAll(loadTraining(False))
    generateSubmissionFileNB()




if __name__ == "__main__": main()
