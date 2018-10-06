import pandas as pd
import scipy as sp
import numpy as np

npzFileName = "ConstructionZone2Vars.npz"

try:
    # True  -> start from scratch
    # False -> load from file
    if False: raise Exception()
    npzFile = np.load(npzFileName)
    
    allClasses              = list(npzFile['allClasses'])
    allClassIDs             = list(npzFile['allClassIDs'])
    allWords                = list(npzFile['allWords'])
    allWordIDs              = list(npzFile['allWordIDs'])
    classMatixes            = list(npzFile['classMatixes'])
    classDocCounts          = list(npzFile['classDocCounts'])
    classProportions        = list(npzFile['classProportions'])
    classWordInstanceCounts = list(npzFile['classWordInstanceCounts'])
    classTotalWordCounts    = list(npzFile['classTotalWordCounts'])
except:
    print("Could not load variables from npz file. Attempting to read raw data.")
    # TODO Print error details.
    try:
        classesFile = open('newsgrouplabels.txt', 'r')
        allClasses = classesFile.read().splitlines()
        classesFile.close()
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    allClassIDs = list(range(1,len(allClasses)+1))
    try:
        vocabFile = open('vocabulary.txt', 'r')
        allWords = vocabFile.read().splitlines()
        vocabFile.close()
    except:
        print("Could not read vocabulary from vocabulary.txt.")
        raise
    allWordIDs = list(range(1,len(allWords)+1))
    
    colNames = ['docID']+allWordIDs+['classID']
    try:
        trainingDf = pd.read_csv('training.csv',header=None,dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
    except:
        print("Could not read training data from training.csv.")
        raise
    docsGroupedByClass = trainingDf.groupby('classID')[allWordIDs]
    classMatixes = [sp.sparse.csr_matrix(docsGroupedByClass.get_group(classID)) for classID in allClassIDs]
    
    trainingDf = None
    docsGroupedByClass = None
    
    classDocCounts = [classMatrix.shape[0] for classMatrix in classMatixes]
    tempSum = sum(classDocCounts)
    classProportions = [classCount/tempSum for classCount in classDocCounts]
    
    classWordInstanceCounts = [ sp.transpose(classMatrix).dot(np.ones(classMatrix.shape[0],dtype=np.int32)) for classMatrix in classMatixes]
    
    classTotalWordCounts = [sum(classWordSums) for classWordSums in classWordInstanceCounts]
    
    np.savez(npzFileName,   allClasses              = allClasses,
                            allClassIDs             = allClassIDs,
                            allWords                = allWords,
                            allWordIDs              = allWordIDs,
                            classMatixes            = classMatixes,
                            classDocCounts          = classDocCounts,
                            classProportions        = classProportions,
                            classWordInstanceCounts = classWordInstanceCounts,
                            classTotalWordCounts    = classTotalWordCounts)
