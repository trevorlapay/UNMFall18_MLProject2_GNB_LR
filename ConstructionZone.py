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


# MAP constants
classColumn = 61189
vocabularyLength = 61188
beta = .01
alpha = 1 + beta
labelsFile = open('newsgrouplabels.txt', "r")
allLabels = labelsFile.read().splitlines()

# word counts won't change (unless we change the training set)
classWordCounts = dict([(1, 154382), (2, 138499), (3, 116141), (4, 103535), (5, 90456),
                        (6, 144656), (7, 64094), (8, 107399), (9, 110928), (10, 124537), (11, 143087),
                        (12, 191242), (13, 97924), (14, 158930), (15, 162521), (16, 236747), (17, 172257),
                        (18, 280067), (19, 172670), (20, 135764)])

# Priors (these won't change unless we change the training set)
priors = [.04025, .052, .051833333333333335, .05358333333333333, .050166666666666665, .0525, .0515, .051166666666666666,
          .05408333333333333, .052333333333333336, .05383333333333333, .05325, .05216666666666667, .05175,
          .05308333333333334, .05425, .04833333333333333, .049416666666666664, .03891666666666667, .035583333333333335]

# Accuracies at various beta levels for plotting.
beta_accuracies = [(.00001, .88367), (.000016, .88367), (.0001, .88957), (.001, .89371), (.01, .89784),
                        (.1, .89193), (.2, .88662), (.5, .87835), (.8, .86625), (1, .85562)]

# Generate a dataframe for priors.
# this should never change, so only do it once and store the values in an array.
def generateMLEpriors(trainingDf):
    # totalClasses = len(allClasses.index)
    numDocs = len(trainingDf)
    priors = []
    for num in range(20):
        prior = len(trainingDf.loc[trainingDf['labelId'] == (num+1)])/numDocs
        priors.append(prior)
    return priors

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
    # drop doc_id column.
    testingDf = testingDf.drop('id', axis=1)
    probabilityMatrix = pd.read_csv("map_probabilities.csv", header=None).drop(0, axis=1).drop(0, axis=0)
    # apply log2 first for calculating log sums
    probabilityMatrix = probabilityMatrix.applymap(math.log2)
    calculated_probabilities = []
    classList = []
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
    listoflists = []
    for numClass, newsGroup in enumerate(allLabels):
        rowdata = []
        totalWordsInClass = classWordCounts.get(numClass + 1)
        trainingDfByClass = trainingDf.loc[trainingDf['labelId'] == (numClass + 1)]
        for numWord, word  in enumerate(allWords):
            if len(trainingDfByClass) > 1:
                # Counting the words in a class is too slow using dataframes.
                # There is probably a faster way to do this (either that, or create a
                # file that just stores these values in a vector or something)
                countWordsinClass = trainingDfByClass[numWord+1].sum()
            else:
                # don't skip zero counts since we are using MAP hallucinated values.
                countWordsinClass = totalWordsInClass = 0
            numerator = countWordsinClass + alpha - 1
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
    testingDf['predictedClass'] = naiveBayesClassify(testingDf)
    listoflists = []
    for numClass, newsgroup1 in enumerate(allLabels):
        row = []
        for numPredictedClass, newsgroup2 in enumerate(allLabels):
            sum_match = len(testingDf.loc[(testingDf['labelId'] == (numClass + 1))
                                                      & (testingDf['predictedClass']
                                                         == (numPredictedClass + 1))])
            row.append(sum_match)
        listoflists.append(row)
    print(listoflists)


#split the training set into both a training and testing set at an index
def splitTrainingTesting(splitAt):
    trainingDfStart = loadTraining()
    trainingDf = trainingDfStart[0:splitAt]
    testingDf = trainingDfStart[splitAt + 1:len(trainingDfStart)]
    return trainingDf, testingDf


def loadTraining():
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    colNames = ['id'] + list(range(1, len(allWords) + 1)) + ['labelId']
    return pd.read_csv("training.csv", header=None,
                             names=colNames).to_sparse(fill_value=0)



def main():
    trainingDf, testingDf = splitTrainingTesting(1200)
    generateMAPmatrix(trainingDf)
    generateConfusionPlot(testingDf)
    # The classifier scores 100% against the training data (I didn't do any Beta-tuning yet)
    # generateSubmissionFileNB()
    # plotBetaValues()


if __name__ == "__main__": main()
