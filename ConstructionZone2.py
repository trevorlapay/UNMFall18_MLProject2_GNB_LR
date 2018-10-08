#%% imports
import pickle
import time
import scipy as sp
import numpy as np
import pandas as pd
import random
import math

pklFileName = "ConstructionZone2Vars.pkl"

def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")

#%% Define data splitting functions.
def getSubset(examples, rows):
    docIDs, mat = examples
    return [docIDs[row] for row in rows], mat[rows,:]

def splitExamples(examples, splitPorportion = 0.5):
    docIDs, mat = examples
    rows = list(range(len(docIDs)))
    random.shuffle(rows)
    splitIndex = round(splitPorportion*len(docIDs))
    return getSubset(examples, rows[:splitIndex]), getSubset(examples, rows[splitIndex:])

def splitClassExamples(classExamples, splitPorportion = 0.5):
    subSet1 = {}
    subSet2 = {}
    for classID, examples in classExamples.items():
        subSet1[classID], subSet2[classID] = splitExamples(examples, splitPorportion)
    return subSet1, subSet2

#%% Load basic data (previously read, dirived and saved).
try:
    # True  -> start from scratch
    # False -> load from file
    if False: raise Exception()
    sTime = time.time()
    with open(pklFileName, 'rb') as pklFile:
        (allClasses,
         allWords,
         allClassExamples,
         testingData,
         classDocCounts,
         classProportions,
         wordCountsInClasses,
         totalWordsInClasses) = pickle.load(pklFile)
        pklFile.close()
        print("Loaded variables in "+str(time.time()-sTime))
except Exception as err:
    print("Could not load variables from pkl file. Error details:")
    print(type(err))
    print(err.args)
    print(err)
    print("\nAttempting to read raw data instead.\nAll runtimes given in seconds.")
    #%% Read in allClasses.
    try:
        sTime = time.time()
        classesFile = open('newsgrouplabels.txt', 'r')
        allClasses = classesFile.read().splitlines()
        classesFile.close()
        allClasses = dict(zip(range(1, len(allClasses)+1), allClasses))
        print("Read allClasses in "+str(time.time()-sTime))
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    #%% Read in allWords.
    try:
        sTime = time.time()
        vocabFile = open('vocabulary.txt', 'r')
        allWords = vocabFile.read().splitlines()
        vocabFile.close()
        allWords = dict(zip(range(1, len(allWords)+1), allWords))
        print("Read allWords in "+str(time.time()-sTime))
    except:
        print("Could not read vocabulary from vocabulary.txt.")
        raise
    #%% Create colNames.
    sTime = time.time()
    colNames = ['docID']+list(allWords.keys())+['classID']
    print("Created colNames in "+str(time.time()-sTime))
    #%% Read trainingDF.
    try:
        sTime = time.time()
        trainingDF = pd.read_csv('training.csv', header=None, dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
        print("Read trainingDF in "+str(time.time()-sTime))
    except:
        print("Could not read training data from training.csv.")
        raise
    #%% Create allClassExamples.
    sTime = time.time()
#    docsGroupedByClass = trainingDF.groupby('classID')[list(allWords.keys())]
#    classMatixes = [sp.sparse.csr_matrix(docsGroupedByClass.get_group(classID))
#                    for classID in allClasses.keys()]
    allClassExamples = {classID : (list(examples['docID']),
                                sp.sparse.csr_matrix(examples[list(allWords.keys())]))
        for classID, examples in trainingDF.groupby('classID')}
    print("Created allClassExamples in "+str(time.time()-sTime))
#    trainingDF = None
#    del trainingDF
#    docsGroupedByClass = None
#    del docsGroupedByClass
    #%% Read testingDF.
    try:
        sTime = time.time()
        testingDF = pd.read_csv('testing.csv', header=None, dtype=np.int32,
                                names=colNames[:-1]).to_sparse(fill_value=0)
        print("Read testingDF in "+str(time.time()-sTime))
    except:
        print("Could not read testing data from testing.csv.")
        raise
    #%% Create testingData
    sTime = time.time()
    testingData = (testingDF['docID'].tolist(), sp.sparse.csr_matrix(testingDF[colNames[1:-1]]))
    print("Created testingData in "+str(time.time()-sTime))
#    colNames = None
#    del colNames
#    testingDF = None
#    del testingDF
    #%% Calculate classProportions.
    sTime = time.time()
    classDocCounts = [len(docIDs) for classID, (docIDs, dataMat) in allClassExamples.items()]
    tempSum = sum(classDocCounts)
    classProportions = [classCount/tempSum for classCount in classDocCounts]
    print("Calculated classProportions in "+str(time.time()-sTime))
#    tempSum = None
#    del tempSum
    #%% Calculate wordCountsInClasses.
    sTime = time.time()
    wordCountsInClasses = [sp.transpose(dataMat).dot(np.ones(len(docIDs), dtype=np.int32))
        for classID, (docIDs, dataMat) in allClassExamples.items()]
    print("Calculated wordCountsInClasses in "+str(time.time()-sTime))
    #%% Calculate totalWordsInClasses.
    sTime = time.time()
    totalWordsInClasses = [sum(classWordCounts) for classWordCounts in wordCountsInClasses]
    print("Calculated totalWordsInClasses in "+str(time.time()-sTime))
    #%% Dump variables.
    sTime = time.time()
    with open(pklFileName, 'wb') as pklFile:
        pickle.dump((allClasses,
                     allWords,
                     allClassExamples,
                     testingData,
                     classDocCounts,
                     classProportions,
                     wordCountsInClasses,
                     totalWordsInClasses), pklFile)
        pklFile.close()
        print("Dumped variables in "+str(time.time()-sTime))

#%% Naive Bayes Learning
sTime = time.time()
#for beta in np.linspace(0,1,1000):
beta = .01 # This is the best preforming beta value Trevor found.
alpha = 1 + beta
temp = (alpha-1)*len(allWords)
mapMmatrix = np.array([(wordCountsInClasses[classID-1]+(alpha-1))/(totalWordsInClasses[classID-1]+temp)
              for classID in allClasses.keys()])
print("Calculated mapMmatrix in "+str(time.time()-sTime))

mapMmatrixLog = np.vectorize(math.log2)(mapMmatrix)

#%% Naive Bayes Testing
likelyhoods = testingData[1].dot(mapMmatrixLog.transpose())
b = np.repeat(np.array([list(allClasses.keys())]), len(likelyhoods), axis=0)
mostLikely = b[np.arange(len(likelyhoods)), np.argmax(likelyhoods, axis=1)]

answersDF = pd.DataFrame()
answersDF['id'] = pd.Series(testingData[0])
answersDF['class'] = pd.Series(mostLikely)
answersDF.to_csv(nowStr()+'answers.csv', index=False)
