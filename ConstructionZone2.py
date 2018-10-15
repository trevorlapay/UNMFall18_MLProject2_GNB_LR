#%% imports
import pickle
import time
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import random
import math

PKL_FILE_NAME = "ConstructionZone2Vars.pkl"
DO_NAIVE_BAYES = False
DO_NAIVE_BAYES_BETA_SEARCHING = False
DO_LOGISTIC_REGRESSION = True

#%% Define time related items
sTime = time.time()
def reportRunTime(taskStr):
    global sTime
    print(taskStr + " in {:.2f} sec".format(time.time()-sTime))
    sTime = time.time()
def nowStr(): return time.strftime("%Y-%m-%d_%H-%M-%S")
#%% Load basic data (previously read, derived and saved).
try:
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
    # ALL_CLASS_EXAMPLES is a dictionary with class IDs as keys and tuples as values. The first item
    # of the tuple is the list of Document IDs matching the class in the key. The second item of the
    # tuple is a matrix where every row corresponds to a document in the list of document IDs and 
    # every column corresponds to a word in the vocabulary. Every cell in this matrix is the count 
    # of the column's word in the row's document.
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
    # TEST_EXAMPLES is structured like one of the tuples in the ALL_CLASS_EXAMPLES dictionary.
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

def splitExamples(examples, splitProportion = 0.5):
    docIDs, dataMat = examples
    assert (len(docIDs) > 1), "Cannot split only one example."
    rows = list(range(len(docIDs)))
    random.shuffle(rows)
    splitIndex = round(splitProportion*len(docIDs))
    if splitIndex >= len(rows)-1: splitIndex -= 1
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
def testClassifier(examples, classifyFunc, **kwargs):
    predictions = classifyFunc(examples[1], **kwargs)
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
    confMatFig.savefig(nowStr()+'confusionMatrix.png')
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
    betaAndErRts = {beta : 0 for beta in np.logspace(start, stop, numBetas)}
    oldBestBeta = 1
    for i in range(1, numDataSplits + 1):
        trainingData, validationData = splitClassExamples(ALL_CLASS_EXAMPLES,0.75)
        for beta, avgErrorRate in betaAndErRts.items():
            mapMat, priors = naiveBayesTrain(trainingData, beta)
            errorRate = validateClassifier(validationData, False, naiveBayesClassify, 
                                           mapMat=mapMat, priors=priors)
            betaAndErRts[beta] = ((i - 1) * avgErrorRate + errorRate) / i
        reportRunTime("Got {} beta error rates for split {}/{}".format(numBetas, i, numDataSplits))
        fileName = "AvgErrorRatesAcross{}BetasOver{}Splits.pkl".format(numBetas,i)
        with open(fileName, 'wb') as betaAndErRtsPklFile:
            pickle.dump(betaAndErRts, betaAndErRtsPklFile)
            betaAndErRtsPklFile.close()
        bestBeta,lowestErrorRate = min(betaAndErRts.items(), key=lambda betaAndErRt: betaAndErRt[1])
        if oldBestBeta != bestBeta:
            print("bestBeta changed from {} to {}".format(oldBestBeta, bestBeta))
            oldBestBeta = bestBeta
        else:
            print("bestBeta stayed "+str(oldBestBeta))
    return betaAndErRts
def plotErrorRatesAccrossBetas(betaAndErRts, fileName="BetaErrorRates"):
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
    avgConfusionMat = np.zeros((len(ALL_CLASSES),len(ALL_CLASSES)), dtype=np.int32)
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
    testClassifier(TEST_EXAMPLES, naiveBayesClassify, mapMat=mapMat, priors=priors)
    reportRunTime("Naive Bayes training, validating and testing")
    if DO_NAIVE_BAYES_BETA_SEARCHING:
        #%% Find best beta
        numBetas = 2000
        numDataSplits = 20
        try:
            for i in range(1, numDataSplits + 1):
                fileName = "AvgErrorRatesAcross{}BetasOver{}Splits.pkl".format(numBetas,i)
                with open(fileName, 'rb') as betaAndErRtsPklFile:
                    betaAndErRts = pickle.load(betaAndErRtsPklFile)
                    betaAndErRtsPklFile.close()
                plotErrorRatesAccrossBetas(betaAndErRts, "BetaErrorRatesAvgOver{}splits".format(i))
        except:
            try:
                fileName = "AvgErrorRatesAcross{}BetasOver{}Splits.pkl".format(numBetas,
                                                                               numDataSplits)
                with open(fileName, 'rb') as betaAndErRtsPklFile:
                    betaAndErRts = pickle.load(betaAndErRtsPklFile)
                    betaAndErRtsPklFile.close()
                plotErrorRatesAccrossBetas(betaAndErRts)
            except:
                betaAndErRts = getBetaErrorRates(numBetas, numDataSplits)
                plotErrorRatesAccrossBetas(betaAndErRts)
#%% Define logistic regression functions
def mashEverythingBackTogether(classExamples):
    deltaMat = sp.sparse.block_diag([np.array([1]*len(docIDs), dtype=np.int8)
                                     for docIDs, dataMat in classExamples.values()])
    wholeDataMat = sp.sparse.vstack([dataMat for docIDs, dataMat in classExamples.values()])
    return wholeDataMat, deltaMat
examplesWithImaginedWord = {classId : (docID,
                                       sp.sparse.hstack([sp.ones((dataMat.shape[0],1)), dataMat],
                                                        'csr', np.int32))
                            for classId, (docID, dataMat) in ALL_CLASS_EXAMPLES.items()}
trainingData, validationData = splitClassExamples(examplesWithImaginedWord, 0.75)
def preprocessData(trainingData):
    mashedTrainingData, deltaMat = mashEverythingBackTogether(trainingData)
    svd = TruncatedSVD(n_components=500)
    dimRedMashedData = svd.fit_transform(mashedTrainingData)
    normingDenominators = np.abs(dimRedMashedData.sum(axis=0)) + 1
    normedDimRedMashedData = dimRedMashedData / normingDenominators
    return svd, normingDenominators, deltaMat, normedDimRedMashedData
def probMat(weightsMat, preprocDataMat):
    preNormed = np.exp(weightsMat * preprocDataMat.transpose())
    normed = preNormed / (preNormed.sum(axis=0)+1)
    return normed
def logisticRegressionTrain(preprocDataMat, deltaMat, numIter, learnRate=0.01, penalty=0.01):
    weightsMat = np.matrix(np.zeros((len(ALL_CLASSES.keys()), preprocDataMat.shape[1])))
    for i in range(numIter):
        weightsMat = weightsMat + learnRate * (
            (deltaMat - probMat(weightsMat, preprocDataMat)) * preprocDataMat
            - penalty * weightsMat)
        if i % 100 == 0:
            reportRunTime("Logistic regression iteration "+str(i))
    return weightsMat
def logisticRegressionClassify(dataMat, svd, normingDenominators, weightsMat):
    if dataMat.shape[1] == len(ALL_WORDS.keys()):
        dataMat = sp.sparse.hstack([sp.ones((dataMat.shape[0],1)), dataMat], 'csr', np.int32)
    dataMat = svd.transform(dataMat)
    dataMat = dataMat / normingDenominators
    likelyhoods = dataMat * weightsMat.transpose()
    return np.array((np.argmax(likelyhoods, axis=1) + 1).flatten().tolist()[0])
if DO_LOGISTIC_REGRESSION:
    #%% Train, validate and test logistic regression
    svd, normingDenominators, deltaMat, preprocDataMat = preprocessData(trainingData)
    errorRates = []
    learnRate = 0.01
    penalty = 0.01
    numIters = np.array([10000])#np.round(np.logspace(3.75, 4.5, 5)).astype(np.int64)
    for numIter in numIters:
        weightsMat = logisticRegressionTrain(preprocDataMat, deltaMat,
                                             numIter=numIter,
                                             learnRate=learnRate, 
                                             penalty=penalty)
        errorRates.append(validateClassifier(validationData, False, logisticRegressionClassify,
                                             svd=svd,
                                             normingDenominators=normingDenominators,
                                             weightsMat=weightsMat))
        print("Error Rate = {:.4f} for {} iterations".format(errorRates[-1], numIter))
        fileName = "LR{}ErrRt{}Iters{}LearnRt{}Penalty.pkl".format(errorRates[-1], numIter,
                                                                   learnRate, penalty)
        with open(fileName, 'wb') as lrTrainingPklFile:
            pickle.dump((svd, normingDenominators, weightsMat), lrTrainingPklFile)
            lrTrainingPklFile.close()
    plt.plot(numIters, np.array(errorRates))
    #plotConfusionMat(confMat, "Confusion Matrix\nError Rate = "+str(errorRate))
    #testClassifier(TEST_EXAMPLES, logisticRegressionClassify, svd=svd,
    #               normingDenominators=normingDenominators, weightsMat=weightsMat)
