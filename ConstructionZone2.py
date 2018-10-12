#%% imports
import pickle
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

PKL_FILE_NAME = "ConstructionZone2Vars.pkl"

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
    with open(PKL_FILE_NAME, 'rb') as pklFile:
        (ALL_CLASSES,
         ALL_WORDS,
         ALL_CLASS_EXAMPLES,
         TEST_EXAMPLES) = pickle.load(pklFile)
        pklFile.close()
        reportRunTime("Loaded variables")
except Exception as err:
    print("Could not load variables from pkl file. Error details:")
    print(type(err))
    print(err.args)
    print(err)
    print("\nAttempting to read raw data instead.\nAll runtimes given in seconds.")
    #%% Read in ALL_CLASSES.
    try:
        classesFile = open('newsgrouplabels.txt', 'r')
        ALL_CLASSES = classesFile.read().splitlines()
        classesFile.close()
        ALL_CLASSES = dict(zip(range(1, len(ALL_CLASSES)+1), ALL_CLASSES))
        reportRunTime("Read ALL_CLASSES")
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    #%% Read in ALL_WORDS.
    try:
        vocabFile = open('vocabulary.txt', 'r')
        ALL_WORDS = vocabFile.read().splitlines()
        vocabFile.close()
        ALL_WORDS = dict(zip(range(1, len(ALL_WORDS)+1), ALL_WORDS))
        reportRunTime("Read ALL_WORDS")
    except:
        print("Could not read vocabulary from vocabulary.txt.")
        raise
    #%% Create colNames.
    colNames = ['docID']+list(ALL_WORDS.keys())+['classID']
    reportRunTime("Created colNames")
    #%% Read trainingDF.
    try:
        trainingDF = pd.read_csv('training.csv', header=None, dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
        reportRunTime("Read trainingDF")
    except:
        print("Could not read training data from training.csv.")
        raise
    #%% Create ALL_CLASS_EXAMPLES.
    ALL_CLASS_EXAMPLES = {classID : (list(examples['docID']),
                                sp.sparse.csr_matrix(examples[list(ALL_WORDS.keys())]))
        for classID, examples in trainingDF.groupby('classID')}
    reportRunTime("Created ALL_CLASS_EXAMPLES")
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
    #%% Create TEST_EXAMPLES
    TEST_EXAMPLES = (testingDF['docID'].tolist(), sp.sparse.csr_matrix(testingDF[colNames[1:-1]]))
    reportRunTime("Created TEST_EXAMPLES")
    colNames = None
    del colNames
    testingDF = None
    del testingDF
    #%% Dump variables.
    with open(PKL_FILE_NAME, 'wb') as pklFile:
        pickle.dump((ALL_CLASSES,
                     ALL_WORDS,
                     ALL_CLASS_EXAMPLES,
                     TEST_EXAMPLES), pklFile)
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
#%% Define general validation and testing functions
def validateClassifier(classExamples, classifyFunction, **kwargs):
    confusionMat = np.zeros((len(ALL_CLASSES),len(ALL_CLASSES)), dtype=np.int32)
    for trueClassID, (docIDs, dataMat) in classExamples.items():
        predictions = classifyFunction(dataMat, **kwargs)
        classConfusion = np.unique(predictions, return_counts=True)
        for predictedClassID, count in zip(list(classConfusion[0]), list(classConfusion[1])):
            confusionMat[trueClassID-1][predictedClassID-1] = count
    return confusionMat
def testClassifier(examples, classifyFunction, **kwargs):
    predictions = classifyFunction(examples[1], **kwargs)
    answersDF = pd.DataFrame()
    answersDF['id'] = pd.Series(examples[0])
    answersDF['class'] = pd.Series(predictions)
    answersDF.to_csv(nowStr()+'answers.csv', index=False)
#%% Define confusion matrix plotting function
def plotConfusionMat(confMat, title="Confusion Matrix"):
    noDiagConfMat = confMat.copy()
    for i in range(len(noDiagConfMat)):
        noDiagConfMat[i, i] = 0
    noDiagConfMat *= -1
    fig, confMatAx = plt.subplots(figsize=(10, 10))
    confMatIm = confMatAx.matshow(noDiagConfMat, cmap=plt.get_cmap("Reds").reversed())
    confMatAx.set_xticks(np.arange(len(ALL_CLASSES)))
    confMatAx.set_yticks(np.arange(len(ALL_CLASSES)))
    confMatAx.set_xticklabels(ALL_CLASSES.values())
    confMatAx.set_yticklabels(ALL_CLASSES.values())
    confMatAx.set_xlabel("Predicted Classes", size=14)
    confMatAx.set_ylabel("True Classes", size=14)
    confMatAx.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(confMatAx.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    textcolors=["black", "white"]
    threshold = confMatIm.norm(noDiagConfMat.max()) / 3
    for i in range(len(ALL_CLASSES)):
        for j in range(len(ALL_CLASSES)):
            if confMat[i, j] != 0:
                if i != j and confMatIm.norm(noDiagConfMat[i,j]) < threshold:
                    color = textcolors[1]
                else:
                    color = textcolors[0]
                confMatAx.text(j, i, confMat[i, j], ha="center", va="center", size=10, color=color)
    confMatAx.set_title(title, size=16)
    fig.tight_layout()
    fig.savefig(nowStr()+'confusionMatrix.png')
    plt.show()
#%% Define Naive Bayes functions
def naiveBayesTrain(classExamples, mapMatAsLog=True):
#    #%% Calculate classProportions.
#    classDocCounts = [len(docIDs) for classID, (docIDs, dataMat) in classExamples.items()]
#    temp = sum(classDocCounts)
#    classProportions = [classCount/temp for classCount in classDocCounts]
    wordCountsInClasses = [sp.transpose(dataMat).dot(np.ones(len(docIDs), dtype=np.int32))
        for classID, (docIDs, dataMat) in classExamples.items()]
    totalWordsInClasses = [sum(classWordCounts) for classWordCounts in wordCountsInClasses]
    #for beta in np.linspace(0,1,1000):
    beta = .01 # This is the best preforming beta value Trevor found.
    temp = (beta)*len(ALL_WORDS)
    mapMat = np.array([(wordCountsInClasses[classID-1]+(beta))
                       /(totalWordsInClasses[classID-1]+temp)
                  for classID in ALL_CLASSES.keys()])
    return np.vectorize(math.log2)(mapMat) if mapMatAsLog else mapMat
def naiveBayesClassify(dataMat, mapMat):
    likelyhoods = dataMat.dot(mapMat.transpose())
    b = np.repeat(np.array([list(ALL_CLASSES.keys())]), len(likelyhoods), axis=0)
    return b[np.arange(len(likelyhoods)), np.argmax(likelyhoods, axis=1)]
#%% Train and validate Naive Bayes with different random selections of tarining and validation sets
metaConfusionMat = np.zeros((len(ALL_CLASSES),len(ALL_CLASSES)), dtype=np.int32)
numValidations = 20
for i in range(numValidations):
    trainingClassExamples, validationClassExamples = splitClassExamples(ALL_CLASS_EXAMPLES, 0.75)
    mapMat = naiveBayesTrain(trainingClassExamples)
    metaConfusionMat += validateClassifier(validationClassExamples, naiveBayesClassify, mapMat=mapMat)
#%% Plot metaConfusionMat
metaConfusionMat = (np.vectorize(round)(metaConfusionMat / numValidations)).astype(np.int32)
plotConfusionMat(metaConfusionMat,
                 "Average Confusion Matrix of "+str(numValidations)+" Naive Bayes Rounds")
#%% Test Naive Bayes
testClassifier(TEST_EXAMPLES, naiveBayesClassify, mapMat=mapMat)
reportRunTime("Naive Bayes training, validating and testing")
