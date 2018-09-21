# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 2018

@author: Luke Hanks and Trevor La Pay
"""

import argparse
import pandas as pd
import math, operator, functools
import scipy.stats as stats


# def naiveBayesClassification():


def generateMLEpriors(trainingDf):
    allClasses = pd.read_csv("newsgrouplabels.txt", header=None)
    # totalClasses = len(allClasses.index)
    numDocs = len(trainingDf)
    priors = []
    for num, val in allClasses.iterrows():
        prior = len(trainingDf.loc[trainingDf[61189] == num])/numDocs
        priors.append(prior)
    return pd.DataFrame(priors)



def generateSparseFiles(trainingDataFile="training.csv"):
    trainingdf = pd.read_csv(trainingDataFile, header=None, nrows=100)
    sparse = trainingdf.to_sparse()
    # this is not sparse - it is loaded with zeros. For now just restricting the number of rows.
    # for now, only generate training (until we get the rest fleshed out)
    sparse.to_csv("sparse_training_tiny.csv")



def main():
    # generateSparseFiles() - we should discuss how to do this. We may not want to use pandas...
    priorsdf = generateMLEpriors(pd.read_csv("sparse_training_tiny.csv", header=None))
    print(priorsdf)


if __name__ == "__main__": main()
