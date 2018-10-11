#%% imports
import pickle
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math


pklFileName = "ConstructionZone2Vars.pkl"

#%% Define time related itmes
sTime = time.time()
def reportRunTime(taskStr):
    global sTime
    print(taskStr + " in "+str(time.time()-sTime))
    sTime = time.time()
def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")
#%% Load basic data (previously read, dirived and saved).
try:
    # True  -> start from scratch
    # False -> load from file
    if False: raise Exception()
    with open(pklFileName, 'rb') as pklFile:
        (allClasses,
         allWords,
         allClassExamples,
         testingData) = pickle.load(pklFile)
        pklFile.close()
        reportRunTime("Loaded variables")
except Exception as err:
    print("Could not load variables from pkl file. Error details:")
    print(type(err))
    print(err.args)
    print(err)
    print("\nAttempting to read raw data instead.\nAll runtimes given in seconds.")
    #%% Read in allClasses.
    try:
        classesFile = open('newsgrouplabels.txt', 'r')
        allClasses = classesFile.read().splitlines()
        classesFile.close()
        allClasses = dict(zip(range(1, len(allClasses)+1), allClasses))
        reportRunTime("Read allClasses")
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    #%% Read in allWords.
    try:
        vocabFile = open('vocabulary.txt', 'r')
        allWords = vocabFile.read().splitlines()
        vocabFile.close()
        allWords = dict(zip(range(1, len(allWords)+1), allWords))
        reportRunTime("Read allWords")
    except:
        print("Could not read vocabulary from vocabulary.txt.")
        raise
    #%% Create colNames.
    colNames = ['docID']+list(allWords.keys())+['classID']
    reportRunTime("Created colNames")
    #%% Read trainingDF.
    try:
        trainingDF = pd.read_csv('training.csv', header=None, dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
        reportRunTime("Read trainingDF")
    except:
        print("Could not read training data from training.csv.")
        raise
    #%% Create allClassExamples.
    allClassExamples = {classID : (list(examples['docID']),
                                sp.sparse.csr_matrix(examples[list(allWords.keys())]))
        for classID, examples in trainingDF.groupby('classID')}
    reportRunTime("Created allClassExamples")
    trainingDF = None
    del trainingDF
    docsGroupedByClass = None
    del docsGroupedByClass
    #%% Read testingDF.
    try:
        testingDF = pd.read_csv('testing.csv', header=None, dtype=np.int32,
                                names=colNames[:-1]).to_sparse(fill_value=0)
        reportRunTime("Read testingDF")
    except:
        print("Could not read testing data from testing.csv.")
        raise
    #%% Create testingData
    testingData = (testingDF['docID'].tolist(), sp.sparse.csr_matrix(testingDF[colNames[1:-1]]))
    reportRunTime("Created testingData")
    colNames = None
    del colNames
    testingDF = None
    del testingDF
    #%% Dump variables.
    with open(pklFileName, 'wb') as pklFile:
        pickle.dump((allClasses,
                     allWords,
                     allClassExamples,
                     testingData), pklFile)
        pklFile.close()
        reportRunTime("Dumped variables")
#%% Define data splitting functions.
def getSubset(examples, rows):
    docIDs, dataMat = examples
    return [docIDs[row] for row in rows], dataMat[rows,:]

def splitExamples(examples, splitPorportion = 0.5):
    docIDs, dataMat = examples
    assert (len(docIDs) > 1), "Cannot split only one example."
    rows = list(range(len(docIDs)))
    random.shuffle(rows)
    splitIndex = round(splitPorportion*len(docIDs))
    if splitIndex >= len(rows)-1: splitIndex -= 1
    return getSubset(examples, rows[:splitIndex]), getSubset(examples, rows[splitIndex:])

def splitClassExamples(classExamples, splitPorportion = 0.5):
    subSet1 = {}
    subSet2 = {}
    for classID, examples in classExamples.items():
        subSet1[classID], subSet2[classID] = splitExamples(examples, splitPorportion)
    return subSet1, subSet2
#%% Seportate allClassExamples into training and validation sets.
trainingClassExamples, validationClassExamples = splitClassExamples(allClassExamples, 0.75)
reportRunTime("Split allClassExamples into trainingClassExamples and validationClassExamples")
#%% Calculate classProportions.
classDocCounts = [len(docIDs) for classID, (docIDs, dataMat) in trainingClassExamples.items()]
temp = sum(classDocCounts)
classProportions = [classCount/temp for classCount in classDocCounts]
reportRunTime("Calculated classProportions")
#%% Calculate wordCountsInClasses.
wordCountsInClasses = [sp.transpose(dataMat).dot(np.ones(len(docIDs), dtype=np.int32))
    for classID, (docIDs, dataMat) in trainingClassExamples.items()]
reportRunTime("Calculated wordCountsInClasses")
#%% Calculate totalWordsInClasses.
totalWordsInClasses = [sum(classWordCounts) for classWordCounts in wordCountsInClasses]
reportRunTime("Calculated totalWordsInClasses")
#%% Naive Bayes Learning
#for beta in np.linspace(0,1,1000):
beta = .01 # This is the best preforming beta value Trevor found.
alpha = 1 + beta
temp = (alpha-1)*len(allWords)
mapMmatrix = np.array([(wordCountsInClasses[classID-1]+(alpha-1))/(totalWordsInClasses[classID-1]+temp)
              for classID in allClasses.keys()])
mapMmatrixLog = np.vectorize(math.log2)(mapMmatrix)
reportRunTime("Calculated mapMmatrix and mapMmatrixLog")
def naiveBayesClassify(dataMat):
    likelyhoods = dataMat.dot(mapMmatrixLog.transpose())
    b = np.repeat(np.array([list(allClasses.keys())]), len(likelyhoods), axis=0)
    return b[np.arange(len(likelyhoods)), np.argmax(likelyhoods, axis=1)]
#%% Naive Bayes Validation
confusionMatrix = np.zeros((len(allClasses),len(allClasses)), dtype=np.int32)
for trueClassID, (docIDs, dataMat) in validationClassExamples.items():
    predictions = naiveBayesClassify(dataMat)
    classConfusion = np.unique(predictions, return_counts=True)
    for predictedClassID, count in zip(list(classConfusion[0]), list(classConfusion[1])):
        confusionMatrix[trueClassID-1][predictedClassID-1] = count
reportRunTime("Calculated predictions for validationClassExamples set")
#%% Plot confusion matrix
noDiagConMat = confusionMatrix.copy()
for i in range(len(noDiagConMat)):
    noDiagConMat[i, i] = 0
noDiagConMat *= -1
fig, conMatAx = plt.subplots(figsize=(8, 8))
conMatIm = conMatAx.matshow(noDiagConMat,cmap=plt.get_cmap("Reds").reversed())
conMatAx.set_xticks(np.arange(len(allClasses)))
conMatAx.set_yticks(np.arange(len(allClasses)))
conMatAx.set_xticklabels(allClasses.keys())
conMatAx.set_yticklabels(allClasses.keys())
conMatAx.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
plt.setp(conMatAx.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
textcolors=["black", "white"]
for i in range(len(allClasses)):
    for j in range(len(allClasses)):
        if confusionMatrix[i, j] != 0:
            if i == j: color = textcolors[0]
            else: color=textcolors[conMatIm.norm(noDiagConMat[i,j])<conMatIm.norm(noDiagConMat.max())/3]
            text = conMatAx.text(j, i, confusionMatrix[i, j],
                           ha="center", va="center", size=10, color=color)
conMatAx.set_title("Confusion Matrix")
fig.tight_layout()
plt.show()
reportRunTime("Plotted confusionMatrix")
#%% Naive Bayes Testing
predictions = naiveBayesClassify(testingData[1])
reportRunTime("Calculated predictions for testingData")
#%% Naive Bayes Testing Submission File
answersDF = pd.DataFrame()
answersDF['id'] = pd.Series(testingData[0])
answersDF['class'] = pd.Series(predictions)
answersDF.to_csv(nowStr()+'answers.csv', index=False)
reportRunTime("Wrote submission file")
