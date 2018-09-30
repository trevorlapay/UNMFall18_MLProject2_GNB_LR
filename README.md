# UNMFall18_MLProject2_GNB_LR
Machine Learning Project 2

# Build instructions

The following files need to be downloaded from kaggle first:
testing.csv
training.csv

The following 5 files are dynamically generated data files. Ensure that the following files are either present or have been generated. the *_split files are optional and are only required if you want to split the training set into testing and training.
label_counts.csv
map_probabilities.csv
priors.csv
training_split.csv
testing_split.csv


If one or more of these do not exist, run the following method in ConstructionZone.py:

generateAll(loadTraining(False)) <— where boolean is whether you want to split the 
training set into supplemental testing. This also creates the split files, which you should comment out if you d not need them. It will take a while. Only generate these files when the training set changes.

If you only need one of those files, run the individual method associated with that file:

generateClassCounts(trainingDf) # Fo label_counts.csv
generatePriors(trainingDf) # for priors.csv
generateMAPmatrix(trainingDf) # for map_probabilities.csv

If you plan on changing the testing set, make sure you re-generate all of the dynamically generated data files.

Assuming you have all of those, you can run the Naive Bayes classifier against the testing.csv by calling def generateSubmissionFileNB():

If you want to see what the beta values plot looks like, run plotBetaValues()

Since we have either SPLIT or WHOLE testing/training data, there are both split and whole versions of those files in the repo. To load the split ones, use loadTrainingAndTestingFromFile(), which assumes you need split. Otherwise, use loadTraining.

If you want to see what the confusion matrix looks like, make sure you have both testing_split and training_split csv-s by running splitTrainingTesting(1200), where 1200 is the index to split the training set into training and testing.



# 