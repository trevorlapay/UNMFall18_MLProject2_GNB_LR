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

# Logistic Regression constants
classColumn = 61189
vocabularyLength = 61188
labelsFile = open('newsgrouplabels.txt', "r")
allLabels = labelsFile.read().splitlines()
learningRate = .001
penaltyTerm = 100

# word counts won't change (unless we change the training set)
# The following 2 constants are ONLY to be used with the entire training set, otherwise generate them from file..
classWordCounts = dict([(1, 154382), (2, 138499), (3, 116141), (4, 103535), (5, 90456),
                        (6, 144656), (7, 64094), (8, 107399), (9, 110928), (10, 124537), (11, 143087),
                        (12, 191242), (13, 97924), (14, 158930), (15, 162521), (16, 236747), (17, 172257),
                        (18, 280067), (19, 172670), (20, 135764)])

# Priors (these won't change unless we change the training set)
priors = [.04025, .052, .051833333333333335, .05358333333333333, .050166666666666665, .0525, .0515, .051166666666666666,
          .05408333333333333, .052333333333333336, .05383333333333333, .05325, .05216666666666667, .05175,
          .05308333333333334, .05425, .04833333333333333, .049416666666666664, .03891666666666667, .035583333333333335]

def generateDeltaMatrix():
    classDf = pd.read_csv("training.csv", header=None).iloc[:,61189]
    #idx = 0
    #new_col = [1] * (len(testingDf) + 1)
    #testingDf.insert(loc=idx, column='x0', value=new_col)
    listoflists = []
    for index, row in enumerate(classDf):
        matrixRow = []
        for index, label in enumerate(allLabels):
            if row == index:
                matrixRow.append(1)
            else:
                matrixRow.append(0)
        listoflists.append(matrixRow)
    print(listoflists)
    pd.DataFrame(listoflists).to_csv("delta_matrix.csv")


def generateSubmissionFileLR(testingDataFile="testing.csv", answersDataFile="answers.csv"):
    vocabFile = open('vocabulary.txt', "r")
    allWords = vocabFile.read().splitlines()
    colNames = ['id'] + list(range(1, len(allWords) + 1))
    testingDF = pd.read_csv(testingDataFile, header=None, names=colNames)
    answerDF = testingDF.filter(['id'], axis=1)
    answerDF['class'] = None #naiveBayesClassify(testingDF)
    answerDF.to_csv(answersDataFile, index=False)

#classify against a testing set and chart how well the algorithm classified.
def generateConfusionPlot(testingDf):
    labelCol = naiveBayesClassify(testingDf)
    testingDf['predictedClass'] = labelCol.values
    testingDf.drop(testingDf.columns.to_series()[1:61189], axis=1, inplace=True)
    listoflists = []
    for numClass, newsgroup1 in enumerate(allLabels):
        row = []
        for numPredictedClass, newsgroup2 in enumerate(allLabels):
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

# regenerate all static data about the training set.
# run this if you change the training set (i.e. run once after splitting the training set at a
# given row index)
def generateAll(trainingDf):
   generatePriors(trainingDf)

def main():
    generateDeltaMatrix()




if __name__ == "__main__": main()
