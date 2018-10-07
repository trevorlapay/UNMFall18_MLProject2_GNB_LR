import pandas as pd
import scipy as sp
import numpy as np
import pickle
import time

pklFileName = "ConstructionZone2Vars.pkl"

try:
    # True  -> start from scratch
    # False -> load from file
    if False: raise Exception()
    
    sTime = time.time()
    with open(pklFileName, 'rb') as pklFile:
        (allClasses,
            allClassIDs,
            allWords,
            allWordIDs,
            classMatixes,
            classDocCounts,
            classProportions,
            wordCountsInClasses,
            totalWordsInClasses,
            testingDocIDs,
            testingMatrix) = pickle.load(pklFile)
        pklFile.close()
        print("Loaded variables in "+str(time.time()-sTime))
except Exception as err:
    print("Could not load variables from npz file. Error details:")
    print(type(err))
    print(err.args)
    print(err)
    print("\nAttempting to read raw data instead.")
    try:
        sTime = time.time()
        classesFile = open('newsgrouplabels.txt', 'r')
        allClasses = classesFile.read().splitlines()
        classesFile.close()
        print("Read allClasses in "+str(time.time()-sTime))
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    allClassIDs = list(range(1,len(allClasses)+1))
    try:
        sTime = time.time()
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
    
    try:
        sTime = time.time()
        trainingDF = pd.read_csv('training.csv',header=None,dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
        print("Read trainingDF in "+str(time.time()-sTime))
    except:
        print("Could not read training data from training.csv.")
        raise
    
    sTime = time.time()
    docsGroupedByClass = trainingDF.groupby('classID')[allWordIDs]
    classMatixes = [sp.sparse.csr_matrix(docsGroupedByClass.get_group(classID)) for classID in allClassIDs]
    print("Created classMatixes in "+str(time.time()-sTime))
    
    trainingDF = None
    del trainingDF
    docsGroupedByClass = None
    del docsGroupedByClass
    
    try:
        sTime = time.time()
        testingDF = pd.read_csv('testing.csv',header=None,dtype=np.int32,
                                 names=colNames[:-1]).to_sparse(fill_value=0)
        print("Read testingDF in "+str(time.time()-sTime))
    except:
        print("Could not read testing data from testing.csv.")
        raise
    
    sTime = time.time()
    testingDocIDs = testingDF['docID'].tolist()
    print("Created testingDocIDs in "+str(time.time()-sTime))
    sTime = time.time()
    testingMatrix = sp.sparse.csr_matrix(testingDF[colNames[1:-1]])
    print("Created testingMatrix in "+str(time.time()-sTime))
    
    colNames = None
    del colNames
    testingDF = None
    del testingDF
    
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
                        totalWordsInClasses,
                        testingDocIDs,
                        testingMatrix], pklFile)
        pklFile.close()
        print("Dumped variables in "+str(time.time()-sTime))


sTime = time.time()
#for beta in np.linspace(0,1,1000):
beta = .01 # This is the best preforming beta value Trevor found.
alpha = 1 + beta
temp = (alpha-1)*len(allWords)
mapMmatrix = [ (wordCountsInClasses[classID-1]+(alpha-1))/(totalWordsInClasses[classID-1]+temp) for classID in allClassIDs ]
print("Calculated mapMmatrix in "+str(time.time()-sTime))


