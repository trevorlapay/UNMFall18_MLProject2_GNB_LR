import pandas as pd
import scipy as sp
import numpy as np
import pickle
import time

npzFileName = "ConstructionZone2Vars.npz"
pklFileName = "ConstructionZone2Vars.pkl"
sTime = time.time()

try:
    # True  -> start from scratch
    # False -> load from file
    if False: raise Exception()
    
    with open(pklFileName, 'rb') as pklFile:
        (allClasses,
            allClassIDs,
            allWords,
            allWordIDs,
            classMatixes,
            classDocCounts,
            classProportions,
            wordCountsInClasses,
            totalWordsInClasses) = pickle.load(pklFile)
        pklFile.close()
        print("Loaded variables in "+str(time.time()-sTime))
except Exception as err:
    print("Could not load variables from npz file. Error details:")
    print(type(err))
    print(err.args)
    print(err)
    print("Attempting to read raw data instead.")
    sTime = time.time()
    try:
        classesFile = open('newsgrouplabels.txt', 'r')
        allClasses = classesFile.read().splitlines()
        classesFile.close()
        print("Read allClasses in "+str(time.time()-sTime))
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    allClassIDs = list(range(1,len(allClasses)+1))
    sTime = time.time()
    try:
        vocabFile = open('vocabulary.txt', 'r')
        allWords = vocabFile.read().splitlines()
        vocabFile.close()
        print("Read allWords in "+str(time.time()-sTime))
    except:
        print("Could not read vocabulary from vocabulary.txt.")
        raise
    sTime = time.time()
    allWordIDs = list(range(1,len(allWords)+1))
    print("Created allWordIDs in "+str(time.time()-sTime))
    
    sTime = time.time()
    colNames = ['docID']+allWordIDs+['classID']
    print("Created colNames in "+str(time.time()-sTime))
    
    sTime = time.time()
    try:
        trainingDf = pd.read_csv('training.csv',header=None,dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
        print("Read trainingDf in "+str(time.time()-sTime))
    except:
        print("Could not read training data from training.csv.")
        raise
    sTime = time.time()
    docsGroupedByClass = trainingDf.groupby('classID')[allWordIDs]
    classMatixes = [sp.sparse.csr_matrix(docsGroupedByClass.get_group(classID)) for classID in allClassIDs]
    print("Created classMatixes in "+str(time.time()-sTime))
    
    colNames = None
    del colNames
    trainingDf = None
    del trainingDf
    docsGroupedByClass = None
    del docsGroupedByClass
    
    sTime = time.time()
    classDocCounts = [classMatrix.shape[0] for classMatrix in classMatixes]
    tempSum = sum(classDocCounts)
    classProportions = [classCount/tempSum for classCount in classDocCounts]
    print("Calculated classProportions in "+str(time.time()-sTime))
    
    tempSum = None
    del tempSum
    
    sTime = time.time()
    wordCountsInClasses = [ sp.transpose(classMatrix).dot(np.ones(classMatrix.shape[0],dtype=np.int32)) for classMatrix in classMatixes]
    print("Calculated wordCountsInClasses in "+str(time.time()-sTime))
    
    sTime = time.time()
    totalWordsInClasses = [sum(classWordCounts) for classWordCounts in wordCountsInClasses]
    print("Calculated totalWordsInClasses in "+str(time.time()-sTime))

    sTime = time.time()
    with open(pklFileName, 'wb') as pklFile:
        pickle.dump([allClasses,
                        allClassIDs,
                        allWords,
                        allWordIDs,
                        classMatixes,
                        classDocCounts,
                        classProportions,
                        wordCountsInClasses,
                        totalWordsInClasses], pklFile)
        pklFile.close()
        print("Dumped variables in "+str(time.time()-sTime))
