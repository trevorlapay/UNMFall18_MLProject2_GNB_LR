#%% imports
import pickle
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd
import random
import math
import csv

BASIC_DATA_PKL_FILE_NAME = "BasicData.pkl"

#%% Decide what to do.
DO_NAIVE_BAYES = False
DO_NAIVE_BAYES_BETA_SEARCHING = False
DO_LOGISTIC_REGRESSION = True
DO_LOGISTIC_REGRESSION_NUM_INTERS_SEARCH = False
DO_LOGISTIC_REGRESSION_LEARN_RATE_SEARCH = True
DO_LOGISTIC_REGRESSION_PENALTY_SEARCH = True
DO_LOGISTIC_REGRESSION_VALIDATE = False
DO_LOGISTIC_REGRESSION_TEST = True

#%% Define time related items
class Timer(object):
    """docstring for Timer"""
    def __init__(self, level=0):
        super(Timer, self).__init__()
        self.startTimes = [time.time()]
        self.level =  max(0, level)
    def setLevel(self, level):
        self.level =  max(0, level)
    def levelUp(self):
        self.level = max(0, self.level - 1)
    def levelDown(self, doReset=True):
        self.level += 1
        if self.level == len(self.startTimes):
            self.startTimes.append(time.time())
        elif doReset:
            self.startTimes[self.level] = time.time()
    def lap(self, taskStr="Did task", doReset=True):
        timeInterval = time.time() - self.startTimes[self.level]
        print("    " * self.level + taskStr + " in {:.2f} sec".format(timeInterval))
        if doReset:
            self.reset()
    def reset(self):
        self.startTimes[self.level] = time.time()
def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")

#%% Define pickle file functions
def savePickle(obj, fileName=None):
    if fileName is None: fileName = nowStr()+"pickleFile.pkl"
    if fileName[-4:] != ".pkl": fileName += ".pkl"
    with open(fileName, 'wb') as pklFile:
        pickle.dump(obj, pklFile)
        pklFile.close()

def loadPickle(fileName="pickleFile.pkl"):
    if fileName[-4:] != ".pkl": fileName += ".pkl"
    obj = None
    with open(fileName, 'rb') as pklFile:
        obj = pickle.load(pklFile)
        pklFile.close()
    return obj

def addToCSV(fields, fileName):
    if fileName[-4:] != ".csv": fileName += ".csv"
    with open(fileName, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

mainTimer = Timer()

#%% Load basic data (previously read, derived and saved).
try:
    (ALL_CLASSES,
     ALL_WORDS,
     ALL_CLASS_EXAMPLES,
     TEST_EXAMPLES) = loadPickle(BASIC_DATA_PKL_FILE_NAME)
    mainTimer.lap("Loaded basic data from pkl file")
except Exception as err:
    print("Could not load basic data from pkl file. Error details:")
    print(type(err))
    print(err.args)
    print(err)
    print("\nAttempting to laod basic data from txt and csv files instead.")
    #%% Read in ALL_CLASSES.
    mainTimer.levelDown()
    try:
        classesFile = open('newsgrouplabels.txt', 'r')
        ALL_CLASSES = classesFile.read().splitlines()
        classesFile.close()
        ALL_CLASSES = dict(zip(range(1, len(ALL_CLASSES)+1), ALL_CLASSES))
        mainTimer.lap("Read ALL_CLASSES")
    except:
        print("Could not read news groups from newsgrouplabels.txt.")
        raise
    #%% Read in ALL_WORDS.
    try:
        vocabFile = open('vocabulary.txt', 'r')
        ALL_WORDS = vocabFile.read().splitlines()
        vocabFile.close()
        ALL_WORDS = dict(zip(range(1, len(ALL_WORDS)+1), ALL_WORDS))
        mainTimer.lap("Read ALL_WORDS")
    except:
        print("Could not read vocabulary from vocabulary.txt.")
        raise
    #%% Create colNames.
    colNames = ['docID']+list(ALL_WORDS.keys())+['classID']
    mainTimer.lap("Created colNames")
    #%% Read trainingDF.
    try:
        trainingDF = pd.read_csv('training.csv', header=None, dtype=np.int32,
                                 names=colNames).to_sparse(fill_value=0)
        mainTimer.lap("Read trainingDF")
    except:
        print("Could not read training data from training.csv.")
        raise
    #%% Create ALL_CLASS_EXAMPLES.
    # ALL_CLASS_EXAMPLES is a dictionary with class IDs as keys and tuples as values. The first item
    # of the tuple is the list of Document IDs matching the class in the key. The second item of the
    # tuple is a matrix where every row corresponds to a document in the list of document IDs and
    # every column corresponds to a word in the vocabulary. Every cell in this matrix is the count
    # of the column's word in the row's document.
    ALL_CLASS_EXAMPLES = {classID : (list(examples['docID']),
                                     sp.sparse.csr_matrix(examples[list(ALL_WORDS.keys())]))
                          for classID, examples in trainingDF.groupby('classID')}
    mainTimer.lap("Created ALL_CLASS_EXAMPLES")
    trainingDF = None
    del trainingDF
    docsGroupedByClass = None
    del docsGroupedByClass
    #%% Read testingDF.
    try:
        testingDF = pd.read_csv('testing.csv', header=None, dtype=np.int32,
                                names=colNames[:-1]).to_sparse(fill_value=0)
        mainTimer.lap("Read testingDF")
    except:
        print("Could not read testing data from testing.csv.")
        raise
    #%% Create TEST_EXAMPLES
    # TEST_EXAMPLES is structured like one of the tuples in the ALL_CLASS_EXAMPLES dictionary.
    TEST_EXAMPLES = (testingDF['docID'].tolist(), sp.sparse.csr_matrix(testingDF[colNames[1:-1]]))
    mainTimer.lap("Created TEST_EXAMPLES")
    colNames = None
    del colNames
    testingDF = None
    del testingDF
    mainTimer.levelUp()
    mainTimer.lap("Loaded basic data from txt and csv files")
    #%% Dump variables.
    savePickle((ALL_CLASSES,
                ALL_WORDS,
                ALL_CLASS_EXAMPLES,
                TEST_EXAMPLES), BASIC_DATA_PKL_FILE_NAME)
    mainTimer.lap("Dumped basic data into pkl file")

#%% Define data splitting functions.
def getSubset(examples, rows):
    docIDs, dataMat = examples
    return [docIDs[row] for row in rows], dataMat[rows,:]

def splitExamples(examples, splitProportion=0.5):
    docIDs, dataMat = examples
    assert (len(docIDs) > 1), "Cannot split only one example."
    rows = list(range(len(docIDs)))
    random.shuffle(rows)
    splitIndex = round(splitProportion*len(docIDs))
    if splitIndex >= len(rows): splitIndex = len(rows)-1
    if splitIndex <= 0: splitIndex = 1
    return getSubset(examples, rows[:splitIndex]), getSubset(examples, rows[splitIndex:])

def splitClassExamples(classExamples, splitProportion = 0.5):
    subSet1 = {}
    subSet2 = {}
    for classID, examples in classExamples.items():
        subSet1[classID], subSet2[classID] = splitExamples(examples, splitProportion)
    return subSet1, subSet2

#%% Define general validation and testing functions
def validateClassifier(classExamples, returnConfMat, classifyFunc, **kwargs):
    if returnConfMat:
        confusionMat = np.zeros((len(ALL_CLASSES),len(ALL_CLASSES)), dtype=np.int32)
    exampleCount = 0
    errorCount = 0
    for trueClassID, (docIDs, dataMat) in classExamples.items():
        predictions = classifyFunc(dataMat, **kwargs)
        classConfusion = np.unique(predictions, return_counts=True)
        for predictedClassID, count in zip(list(classConfusion[0]), list(classConfusion[1])):
            if returnConfMat:
                confusionMat[trueClassID-1][predictedClassID-1] = count
            exampleCount += count
            if predictedClassID != trueClassID:
                errorCount += count
    errorRate = errorCount/exampleCount
    if returnConfMat: return errorRate, confusionMat
    else: return errorRate

def testClassifier(examples, fileNamePrefix, classifyFunc, **kwargs):
    docIDs, dataMat = examples
    predictions = classifyFunc(dataMat, **kwargs)
    answersDF = pd.DataFrame()
    answersDF['id'] = pd.Series(docIDs)
    answersDF['class'] = pd.Series(predictions)
    answersDF.to_csv(nowStr()+fileNamePrefix+'Answers.csv', index=False)

#%% Define confusion matrix plotting function
def plotConfusionMat(confMat, title="Confusion Matrix"):
    noDiagConfMat = confMat.copy()
    for i in range(len(noDiagConfMat)):
        noDiagConfMat[i, i] = 0
    noDiagConfMat *= -1
    confMatFig, confMatAx = plt.subplots(figsize=(10, 10))
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
                confMatAx.text(j, i, confMat[i,j], ha="center", va="center", size=10, color=color)
    confMatAx.set_title(title, size=16)
    confMatFig.tight_layout()
    confMatFig.savefig(nowStr()+'ConfusionMatrix.png')
    plt.show()

#%% Define Naive Bayes functions
def naiveBayesTrain(classExamples, beta=0.0267, asLog=True):
    classDocCounts = [len(docIDs) for classID, (docIDs, dataMat) in classExamples.items()]
    temp = sum(classDocCounts)
    priors = np.array([classCount/temp for classCount in classDocCounts])
    wordCountsInClasses = [sp.transpose(dataMat).dot(np.ones(len(docIDs), dtype=np.int32))
                           for classID, (docIDs, dataMat) in classExamples.items()]
    totalWordsInClasses = [sum(classWordCounts) for classWordCounts in wordCountsInClasses]
    temp = (beta)*len(ALL_WORDS)
    mapMat = np.array([(wordCountsInClasses[classID-1]+(beta))
                       /(totalWordsInClasses[classID-1]+temp)
                       for classID in ALL_CLASSES.keys()])
    if asLog:
        return np.vectorize(math.log2)(mapMat), np.vectorize(math.log2)(priors)
    else:
        return mapMat, priors

def naiveBayesClassify(dataMat, mapMat, priors):
    likelyhoods = dataMat.dot(mapMat.transpose()) + priors
    b = np.repeat(np.array([list(ALL_CLASSES.keys())]), len(likelyhoods), axis=0)
    return b[np.arange(len(likelyhoods)), np.argmax(likelyhoods, axis=1)]

#%% Define beta error rate functions
def getBetaErrorRates(numBetas=100, numDataSplits=10, start=-5, stop=0):
    mainTimer.levelDown()
    betaAndErRts = {beta : 0 for beta in np.logspace(start, stop, numBetas)}
    oldBestBeta = 1
    for i in range(1, numDataSplits + 1):
        mainTimer.reset()
        trainingData, validationData = splitClassExamples(ALL_CLASS_EXAMPLES, 0.75)
        for beta, avgErrorRate in betaAndErRts.items():
            mapMat, priors = naiveBayesTrain(trainingData, beta)
            errorRate = validateClassifier(validationData, False, naiveBayesClassify,
                                           mapMat=mapMat, priors=priors)
            betaAndErRts[beta] = ((i - 1) * avgErrorRate + errorRate) / i
        mainTimer.lap("Got {} beta error rates for split {}/{}".format(numBetas, i, numDataSplits))
        fileName = "AvgErrorRatesAcross{}BetasOver{}Splits".format(numBetas,i)
        savePickle(betaAndErRts, fileName)
        bestBeta,lowestErrorRate = min(betaAndErRts.items(), key=lambda betaAndErRt: betaAndErRt[1])
        if oldBestBeta != bestBeta:
            print("bestBeta changed from {} to {}".format(oldBestBeta, bestBeta))
            oldBestBeta = bestBeta
        else:
            print("bestBeta stayed "+str(oldBestBeta))
    mainTimer.levelUp()
    return betaAndErRts

def plotErrorRatesAcrossBetas(betaAndErRts, fileName="BetaErrorRates"):
    betaErRtFig, betaErRtAx = plt.subplots(figsize=(10, 10))
    betaErRtIm = betaErRtAx.plot(*zip(*(betaAndErRts.items())))
    betaErRtAx.set_xscale('log')
    betaErRtAx.set_xlabel("Beta")
    betaErRtAx.set_ylabel("Average Error Rate")
    betaErRtAx.set_title("Error Rates Across Beta Values", size=14)
    bestBeta, lowestErrorRate = min(betaAndErRts.items(), key=lambda betaAndErRt : betaAndErRt[1])
    text = "beta={:.4f}, error rate={:.4f}".format(bestBeta, lowestErrorRate)
    bboxProps = dict(boxstyle="square,pad=0.2", fc="w", lw=0.5)
    arrowProps = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    betaErRtAx.annotate(text, xy=(bestBeta, lowestErrorRate), xytext=(0.6,0.025), xycoords='data',
                        textcoords="axes fraction", arrowprops=arrowProps, bbox=bboxProps,
                        ha="right", va="top")
    betaErRtFig.tight_layout()
    betaErRtFig.savefig(nowStr()+fileName+".png")
    betaErRtFig.tight_layout()
    plt.show()

if DO_NAIVE_BAYES:
    #%% Train and validate Naive Bayes with random subsets
    avgErrorRate = 0
    avgConfusionMat = np.zeros((len(ALL_CLASSES),len(ALL_CLASSES)), dtype=np.int64)
    numDataSplits = 20
    for i in range(numDataSplits):
        trainingData, validationData = splitClassExamples(ALL_CLASS_EXAMPLES, 0.75)
        mapMat, priors = naiveBayesTrain(trainingData)
        errorRate, confusionMat = validateClassifier(validationData, True, naiveBayesClassify,
                                                     mapMat=mapMat, priors=priors)
        avgErrorRate += errorRate
        avgConfusionMat += confusionMat
    avgErrorRate = round(avgErrorRate/numDataSplits, 4)
    #%% Plot avgConfusionMat
    avgConfusionMat = (np.vectorize(round)(avgConfusionMat / numDataSplits)).astype(np.int32)
    plotConfusionMat(avgConfusionMat,
                     "Average Confusion Matrix of "+str(numDataSplits)+" Naive Bayes Rounds"
                     +"\nAverage Error Rate = "+str(avgErrorRate))
    #%% Test Naive Bayes
    mapMat, priors = naiveBayesTrain(ALL_CLASS_EXAMPLES)
    testClassifier(TEST_EXAMPLES, 'NB', naiveBayesClassify, mapMat=mapMat, priors=priors)
    mainTimer.lap("Naive Bayes training, validating and testing")
    
    if DO_NAIVE_BAYES_BETA_SEARCHING:
        #%% Find best beta
        numBetas = 2000
        numDataSplits = 20
        try:
            fileName = "AvgErrorRatesAcross{}BetasOver{}Splits".format(numBetas, numDataSplits)
            betaAndErRts = loadPickle(fileName)
            plotErrorRatesAcrossBetas(betaAndErRts)
        except:
            mainTimer.reset()
            betaAndErRts = getBetaErrorRates(numBetas, numDataSplits)
            plotErrorRatesAcrossBetas(betaAndErRts)
            mainTimer.lap("Found best beta")

#%% Define logistic regression functions
def mashEverythingBackTogether(classExamples):
    deltaMat = sp.sparse.block_diag([np.array([1]*len(docIDs), dtype=np.int8)
                                     for docIDs, dataMat in classExamples.values()])
    wholeDataMat = sp.sparse.vstack([dataMat for docIDs, dataMat in classExamples.values()])
    return wholeDataMat, deltaMat

class AntiDimRed(object):
    """docstring for AntiDimRed"""
    def __init__(self, arg=None):
        super(AntiDimRed, self).__init__()
        self.arg = arg
    def transform(self, dataMat):
        return dataMat
    def fit_transform(self, dataMat):
        return dataMat

def prependOnesColumn(dataMat):
    if sp.sparse.issparse(dataMat):
        return sp.sparse.hstack([sp.ones((dataMat.shape[0],1)), dataMat], 'csr', np.int32)
    else:
        return np.hstack((np.ones((dataMat.shape[0],1)), dataMat))

def preprocessTrainingData(trainingData, reduceDimTo=0):
    dataMat, deltaMat = mashEverythingBackTogether(trainingData)
    mainTimer.reset()
    mainTimer.levelDown()
    if reduceDimTo >= 2:
        reducer = skl.decomposition.TruncatedSVD(n_components=reduceDimTo)
    else:
        reducer = AntiDimRed()
    dataMat = prependOnesColumn(reducer.fit_transform(dataMat))
    mainTimer.lap("Reduced dimensionality for LR")
    normingDivisors = np.abs(dataMat.sum(axis=0)) + 1
    mainTimer.lap("Calculated normalizing divisors for LR")
    dataMat = dataMat / normingDivisors
    mainTimer.lap("Normalized data for LR")
    mainTimer.levelUp()
    return reducer, normingDivisors, deltaMat, dataMat

def probMat(weightsMat, dataMat):
    preNormed = np.exp(weightsMat * dataMat.transpose())
    normed = preNormed / (preNormed.sum(axis=0)+1)
    return normed

def logisticRegressionTrain(dataMat, deltaMat, learnRate=0.01, penalty=0.01, numIter=10):
    reportingInterval = int(round(numIter/10))
    weightsMat = np.matrix(np.zeros((len(ALL_CLASSES.keys()), dataMat.shape[1])))
    mainTimer.levelDown()
    for i in range(round(numIter)):
        weightsMat = weightsMat + learnRate * (
            (deltaMat - probMat(weightsMat, dataMat)) * dataMat
            - penalty * weightsMat)
        if reportingInterval <= 1 or (i+1) % reportingInterval == 0:
            mainTimer.lap("LR iterations {} to {}".format(i+1-reportingInterval,i))
    mainTimer.levelUp()
    return weightsMat

def preprocessUnclassifiedData(dataMat, reducer, normingDivisors):
    return prependOnesColumn(reducer.transform(dataMat)) / normingDivisors

def logisticRegressionClassify(dataMat, reducer, normingDivisors, weightsMat):
    dataMat = preprocessUnclassifiedData(dataMat, reducer, normingDivisors)
    likelyhoods = dataMat * weightsMat.transpose()
    return np.array((np.argmax(likelyhoods, axis=1) + 1).flatten().tolist()[0])

if DO_LOGISTIC_REGRESSION:
    #%% Train, validate and test logistic regression
    reduceDimTo = 0 # 0 indicates not to reduce dimensions at all.
    learnRate = 0.01
    penalty = 0.001
    numIter = 100
    if (DO_LOGISTIC_REGRESSION_NUM_INTERS_SEARCH or
        DO_LOGISTIC_REGRESSION_LEARN_RATE_SEARCH or
        DO_LOGISTIC_REGRESSION_PENALTY_SEARCH or
        DO_LOGISTIC_REGRESSION_VALIDATE):
        trainingData, validationData = splitClassExamples(ALL_CLASS_EXAMPLES, 0.75)
        mainTimer.lap("Split data into training and validation")
        reducer, normingDivisors, deltaMat, dataMat = preprocessTrainingData(trainingData, reduceDimTo)
        mainTimer.lap("LR preprocessing")
    weightsMat = None
    errorRates = []
    
    if DO_LOGISTIC_REGRESSION_NUM_INTERS_SEARCH:
        mainTimer.reset()
        print("numDims = {} learnRate = {}  penalty = {}".format(dataMat.shape[1],learnRate,
                                                                 penalty))
        if dataMat.shape[1] < 10000:
            iterNums = np.round(np.logspace(3, 5, 3)).astype(np.int64)
        else:
            iterNums = np.round(np.logspace(0, 3, 4)).astype(np.int64)
        mainTimer.levelDown()
        for numIter in iterNums:
            weightsMat = logisticRegressionTrain(dataMat, deltaMat,
                                                 learnRate=learnRate,
                                                 penalty=penalty,
                                                 numIter=numIter)
            mainTimer.lap("Trained LR for {} interations".format(numIter))
            
            errorRates.append(validateClassifier(validationData, False, logisticRegressionClassify,
                                                 reducer=reducer,
                                                 normingDivisors=normingDivisors,
                                                 weightsMat=weightsMat))
            mainTimer.lap("Validated LR")
            addToCSV([errorRates[-1], dataMat.shape[1], learnRate, penalty, numIter],
                     'LR_Validations.csv')
            print("Error Rate = {:.4f}".format(errorRates[-1], numIter))
        mainTimer.levelUp()
        plt.plot(iterNums, np.array(errorRates))
        mainTimer.lap("Found best number of iterations for LR")
    
    if DO_LOGISTIC_REGRESSION_LEARN_RATE_SEARCH:
        mainTimer.reset()
        mainTimer.lap("Found best learning rate for LR")
    
    if DO_LOGISTIC_REGRESSION_PENALTY_SEARCH:
        print("Starting best penalty ")
        mainTimer.reset()
        mainTimer.lap("Found best penalty strength for LR")
    
    if DO_LOGISTIC_REGRESSION_VALIDATE:
        mainTimer.reset()
        weightsMat = logisticRegressionTrain(dataMat, deltaMat,
                                             learnRate=learnRate,
                                             penalty=penalty,
                                             numIter=numIter)
        mainTimer.lap("Trained LR")
        errorRate, confusionMat = validateClassifier(validationData, True,
                                                     logisticRegressionClassify,
                                                     reducer=reducer,
                                                     normingDivisors=normingDivisors,
                                                     weightsMat=weightsMat)
        print("Error Rate = {:.4f} for {} iterations".format(errorRate, numIter))
        plotConfusionMat(confusionMat, "Confusion Matrix\nError Rate = "+str(errorRate))
        addToCSV([errorRate, dataMat.shape[1], learnRate, penalty, numIter], 'LR_Validations.csv')
        mainTimer.lap("Validated LR")
    if DO_LOGISTIC_REGRESSION_TEST:
        numDims = (reduceDimTo if reduceDimTo >= 2 else ALL_CLASS_EXAMPLES[1][1].shape[1]) + 1
        fileName = "LR_{}Dims_{}LearnRt_{}Penalty_{}Iters".format(numDims,
                                                                  learnRate,
                                                                  penalty,
                                                                  numIter)
        try:
            (reducer, normingDivisors, weightsMat) = loadPickle(fileName)
        except:
            reducer, normingDivisors, deltaMat, dataMat = preprocessTrainingData(ALL_CLASS_EXAMPLES,
                                                                                 reduceDimTo)
            weightsMat = logisticRegressionTrain(dataMat, deltaMat,
                                                learnRate=learnRate,
                                                penalty=penalty,
                                                numIter=numIter)
            fileName = "LR_{}Dims_{}LearnRt_{}Penalty_{}Iters".format(dataMat.shape[1],
                                                                    learnRate,
                                                                    penalty,
                                                                    numIter)
            savePickle((reducer, normingDivisors, weightsMat), fileName)
        testClassifier(TEST_EXAMPLES, 'LR', logisticRegressionClassify, reducer=reducer,
                       normingDivisors=normingDivisors, weightsMat=weightsMat)
        mainTimer.lap("Tested LR")
