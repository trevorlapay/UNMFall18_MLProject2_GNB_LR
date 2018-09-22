import argparse as ap
import pandas as pd
import scipy as sp
import numpy as np

#pd.read_csv("training.csv", header=None, nrows=200).to_csv("sparse_training_tiny.csv",header=False)

labelsFile = open('newsgrouplabels.txt', "r")
allLabels = labelsFile.read().splitlines()
labelsFile.close()
numLabels = len(allLabels)
allLabelIds = list(range(1,numLabels+1))

vocabFile = open('vocabulary.txt', "r")
allWords = vocabFile.read().splitlines()
vocabFile.close()
numWords = len(allWords)
allWordIds = list(range(1,numWords+1))

colNames = ['docId']+list(range(1,len(allWords)+1))+['labelId']
trainingDf = pd.read_csv("training.csv",header=None,
                         names=colNames).to_sparse(fill_value=0)
print(trainingDf)

labelProportions = trainingDf['labelId'].value_counts(normalize=True)
print(labelProportions)
labelProportions.plot(kind='pie')

labelWordSums = trainingDf.groupby('labelId')[allWordIds].agg(np.sum).to_sparse(fill_value=0)
print(labelWordSums)

labelSums = labelWordSums.sum(axis=1)
print(labelSums)
labelSums.plot(kind='bar')
