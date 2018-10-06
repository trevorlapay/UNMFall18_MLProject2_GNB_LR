import pandas as pd
import scipy as sp
import numpy as np

try:
    # True  -> start from scratch
    # False -> load from file
    if False: raise Exception()
    npzFile = np.load('ConstructionZone2Vars.npz')
    
    allLabels = list(npzFile['allLabels'])
    numLabels = npzFile['numLabels'].item()
    allLabelIds = list(npzFile['allLabelIds'])
    allWords = list(npzFile['allWords'])
    numWords = npzFile['numWords'].item()
    allWordIds = list(npzFile['allWordIds'])
    listOfLabelMatixes = list(npzFile['listOfLabelMatixes'])
    labelCounts = list(npzFile['labelCounts'])
    labelProportions = list(npzFile['labelProportions'])
    labelsWordSums = list(npzFile['labelsWordSums'])
    labelSums = list(npzFile['labelSums'])
except:
    labelsFile = open('newsgrouplabels.txt', 'r')
    allLabels = labelsFile.read().splitlines()
    labelsFile.close()
    numLabels = len(allLabels)
    allLabelIds = list(range(1,numLabels+1))
    
    vocabFile = open('vocabulary.txt', 'r')
    allWords = vocabFile.read().splitlines()
    vocabFile.close()
    numWords = len(allWords)
    allWordIds = list(range(1,numWords+1))
    
    colNames = ['docId']+list(range(1,len(allWords)+1))+['labelId']
    trainingDf = pd.read_csv('training.csv',header=None,dtype=np.int32,
                             names=colNames).to_sparse(fill_value=0)
    print(trainingDf)
    
    docsGroupedByLabel = trainingDf.groupby('labelId')[allWordIds]
    
    listOfLabelMatixes = [sp.sparse.csr_matrix(docsGroupedByLabel.get_group(labelID)) for labelID in allLabelIds]
    
    labelCounts = [labelMatrix.shape[0] for labelMatrix in listOfLabelMatixes]
    tempSum = sum(labelCounts)
    labelProportions = [labelCount/tempSum for labelCount in labelCounts]
    
    labelsWordSums = [ sp.transpose(labelMatrix).dot(np.ones(labelMatrix.shape[0],dtype=np.int32)) for labelMatrix in listOfLabelMatixes]
    
    labelSums = [sum(labelWordSums) for labelWordSums in labelsWordSums]
    
    np.savez('ConstructionZone2Vars.npz',allLabels = allLabels,
                                        numLabels = numLabels,
                                        allLabelIds = allLabelIds,
                                        allWords = allWords,
                                        numWords = numWords,
                                        allWordIds = allWordIds,
                                        listOfLabelMatixes = listOfLabelMatixes,
                                        labelCounts = labelCounts,
                                        labelProportions = labelProportions,
                                        labelsWordSums = labelsWordSums,
                                        labelSums = labelSums)
