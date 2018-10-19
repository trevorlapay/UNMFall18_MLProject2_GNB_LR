# UNMFall18_MLProject2_GNB_LR
Machine Learning Project 2
Luke Hanks
Trevor La Pay
Kyle Jurney

# Build instructions

The following files need to be downloaded from kaggle first:
testing.csv
training.csv

LOADING DATA:
The code tries to load the basic data (ALL_CLASSES, ALL_WORDS, ALL_CLASS_EXAMPLES, TEST_EXAMPLES) 
from a pickle file called "ConstructionZone2Vars.pkl" which takes less than a second. 
If that doesn't work, then it will try to load the data from the original files (found on Kaggle)
which takes about 30 minutes. If the data is loaded from the original files, the script will 
create the "ConstructionZone2Vars.pkl" pickle file for next time.

To run Naive Bayes classification, Logistic Regression, or any combination of those and the
sub-problems we solved in this project, find the boolean flags at page top of 
the file ML_2018_NaiveBayes_LogisticRegression.py and modify what you want to run to True:

#%% Decide what to do.
DO_NAIVE_BAYES = False
DO_NAIVE_BAYES_BETA_SEARCHING = False
DO_LOGISTIC_REGRESSION = True
DO_LOGISTIC_REGRESSION_NUM_INTERS_SEARCH = False
DO_LOGISTIC_REGRESSION_LEARN_RATE_SEARCH = False
DO_LOGISTIC_REGRESSION_PENALTY_SEARCH = False
DO_LOGISTIC_REGRESSION_VALIDATE = True
DO_LOGISTIC_REGRESSION_TEST = True



# 