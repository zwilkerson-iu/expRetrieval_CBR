from case_base import CaseBase
from case import Case
from reader import Reader
from feature import Feature
from deepNetwork import DeepImageNetwork

from skimage.io import imread
import numpy as np
import tensorflow as tf
import os
import sys
import random
import statistics

#TODO: documentation here!
def majorityRule(caseBase:CaseBase, neighbors:list, mode:str = "majority"):
    if type(caseBase.retrieveCase(neighbors[0][0]).result) is tuple:
        regressionFlag = False
    else:
        regressionFlag = True
    resultOptions = {}
    if mode == "majority":
        for caseID, _ in neighbors:
            resultTemp = caseBase.retrieveCase(caseID).result
            if resultOptions.get(resultTemp) is not None:
                if not regressionFlag:
                    resultOptions[resultTemp[0]][0] += resultTemp[1]
                    resultOptions[resultTemp[0]][1] += 1
                else:
                    resultOptions[resultTemp][0] += resultTemp
                    resultOptions[resultTemp][1] += 1
            else:
                if not regressionFlag:
                    resultOptions[resultTemp[0]] = (resultTemp[1], 1)
                else:
                    resultOptions[resultTemp] = (resultTemp, 1)
        
        majorityResult = None
        strongestVote = -sys.maxsize
        for resultValue, (resultSum, numberOfVotes) in resultOptions.items():
            if resultSum > strongestVote:
                strongestVote = resultSum
                majorityResult = resultValue
        if not regressionFlag:
            return (majorityResult, strongestVote / numberOfVotes)
        else:
            return resultValue

    elif mode == "weighted": #TODO: implement weights (especially helpful for AwA2 cases)
        print("TODO: implement this")
    else:
        print("Error: invalid mode parameter")


"""
Performs k-fold cross-validation on the provided case base, using a KNN algorithm
- caseBase = the parent case base, which is divided into subsets for training and testing
- numberOfFolds = the number of folds for testing purposes
- k = the number of neighbors to consider in the KNN algorithm
Returns: a list of accuracies for each fold in the cross-validation process (i.e., the length of the list
            will be the same as the number of folds)
"""
def kFoldCrossValidation(caseBase:CaseBase, numberOfFolds:int, k:int = 5, dynamicCaseBase:bool = False):
    foldSize = int(caseBase.caseBaseSize/numberOfFolds)
    caseID_master = list(caseBase.cases.keys())
    accuracies = []
    for kFold in range(numberOfFolds):
        testingCases = caseID_master[kFold*foldSize:(kFold+1)*foldSize]
        trainingCases = caseID_master[:kFold*foldSize] + caseID_master[(kFold+1)*foldSize:]
        tempCaseBase = CaseBase()
        for caseID in trainingCases:
            tempCaseBase.addCase(caseBase.retrieveCase(caseID))
        correctClassifications = 0.0
        for caseID in testingCases:
            neighbors = tempCaseBase.getKClosestCases(caseBase.retrieveCase(caseID), k)
            queryResult = majorityRule(tempCaseBase, neighbors)
            temp = Case(caseBase.retrieveCase(caseID).getFeatures(), queryResult)
            if dynamicCaseBase:
                tempCaseBase.addCase(temp)
            if temp.result == caseBase.retrieveCase(caseID).result: #May need to change this depending on adaptation/confidence strategy
                correctClassifications += 1
            else:
                print("Incorrect result:", temp.result, "(" + str(caseBase.retrieveCase(caseID).result) + ")")
        accuracies.append(correctClassifications/len(testingCases))
    return accuracies

"""
Special case of k-fold cross-validation where the number of folds is equal to the size of the case base
- k = the number of neighbors to consider in the KNN algorithm
Returns: a list of accuracies for each fold in the cross-validation process (i.e., the length of the list
            will be the same as the number of folds)
"""
def leaveOneOut(caseBase:CaseBase, k:int = 1):
    return kFoldCrossValidation(caseBase, caseBase.caseBaseSize, k)

"""
Test controller for evaluating the accuracy of the case base using partial feature queries (dictated by the partiality
    threashold percentage).
- caseBase = the case base used for testing
- numberOfIterations = the number of query cases to use (randomly selected from the case base, with features selected
                        randomly relative to the partiality threshold)
- partialityThreshold = the percentage of features that should be considered for each query case (features are selected
                        randomly, based on an associated value on [0.0,1.0] that is compared against this value) OR the
                        number of features to use for testing (if >= 1)
- k = the number of neighbors to consider in the KNN algorithm
Returns: the accuracy percentage of the case base over all tests as a decimal
"""
def partialFeatureValidation(caseBase:CaseBase, numberOfIterations:int, partialityThreshold:float = 0.1, k:int = 1):
    correctClassifications = 0
    for _ in range(numberOfIterations):
        caseIndex = random.randint(0, caseBase.caseBaseSize - 1)
        case = caseBase.retrieveCase(list(caseBase.cases.keys())[caseIndex])
        temp = Case({}, None)
        if partialityThreshold < 1:
            for featureName in case.features.keys():
                if random.random() < partialityThreshold:
                    temp.addNewFeature(case.features[featureName])
        else:
            for featureName in random.sample(list(case.features.keys()), partialityThreshold):
                temp.addNewFeature(case.features[featureName])
        neighbors = caseBase.getKClosestCases(temp, k)
        queryResult = majorityRule(caseBase, neighbors)
        temp.result = queryResult
        if temp.result == case.result:
            correctClassifications += 1
    return correctClassifications / float(numberOfIterations)

"""
Test controller for evaluating the accuracy of the case base using removed-case queries.  This is only to be used
    when each image used for feature learning gets its corresponding features assigned to an individual case
- caseBase = the case base used for testing
- numberOfIterations = the number of query cases to use (each randomly selected from the case base)
- k = the number of neighbors to consider in the KNN algorithm
Returns: the accuracy percentage of the case base over all tests as a decimal
"""
def duplicatedFeatureValidation(caseBase:CaseBase, numberOfIterations:int, k:int = 2):
    correctClassifications = 0
    for _ in range(numberOfIterations):
        caseIndex = random.randint(0, caseBase.caseBaseSize - 1)
        case = caseBase.retrieveCase(list(caseBase.cases.keys())[caseIndex])
        # caseBase.removeCase(list(caseBase.cases.keys())[caseIndex])
        neighbors = caseBase.getKClosestCases(case, k)
        queryResult = majorityRule(caseBase, neighbors[1:])
        if queryResult == case.result:
            correctClassifications += 1
        # caseBase.addCase(case)
    return correctClassifications / float(numberOfIterations)

#TODO: documentation here!
def generateImageSample(numImagesPerAnimal:int, rootDir:str):
        images = []
        classes = os.listdir(rootDir + "awa2/JPEGImages")
        for animal in classes:
            imageFiles = os.listdir(rootDir + "awa2/JPEGImages/" + animal)
            imageTemps = random.sample(imageFiles, numImagesPerAnimal)
            for filepath in imageTemps:
                temp = imread(rootDir + "awa2/JPEGImages/" + animal + "/" + filepath, as_gray = False)
                images.append(temp)
        return images

#TODO: documentation here!
def generateCaseListWithLearnedFeatures(learnedFeatures, numImagesPerAnimal:int, rootDir:str, useExpertFeatures:bool = True, matchTest = True):
    cases = []
    classes = os.listdir(rootDir + "awa2/JPEGImages")
    if matchTest:
        if useExpertFeatures:
            for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-continuous.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
                index = classes.index(case.result[0])
                for i in range(len(learnedFeatures[0])):
                    sumTemp = 0.0
                    for j in range(numImagesPerAnimal):
                        sumTemp += learnedFeatures[index * numImagesPerAnimal + j][i]
                    case.addNewFeature(Feature("Feature" + str(i), sumTemp / numImagesPerAnimal, 1, "euclideanDistance"))
                cases.append(case)
        else:
            for caseName in classes:
                temp = Case({}, (caseName, 1.0))
                index = classes.index(caseName)
                for i in range(len(learnedFeatures[0])):
                    sumTemp = 0.0
                    for j in range(numImagesPerAnimal):
                        sumTemp += learnedFeatures[index * numImagesPerAnimal + j][i]
                    temp.addNewFeature(Feature("Feature" + str(i), sumTemp / numImagesPerAnimal, 1, "euclideanDistance"))
                cases.append(temp)
    else: #removalTest
        if useExpertFeatures:
            for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-continuous.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
                index = classes.index(case.result[0])
                for j in range(numImagesPerAnimal):
                    temp = Case({}, (case.result[0], 1.0))
                    for feature in case.features.values():
                        temp.addNewFeature(feature)
                    for i in range(len(learnedFeatures[0])):
                        temp.addNewFeature(Feature("Feature" + str(i), float(learnedFeatures[index * numImagesPerAnimal + j][i]), 1, "euclideanDistance"))
                    cases.append(temp)
        else:
            for caseName in classes:
                index = classes.index(caseName)
                for j in range(numImagesPerAnimal):
                    temp = Case({}, (caseName, 1.0))
                    for i in range(len(learnedFeatures[0])):
                        temp.addNewFeature(Feature("Feature" + str(i), float(learnedFeatures[index * numImagesPerAnimal + j][i]), 1, "euclideanDistance"))
                    cases.append(temp)
    return cases

#TODO: documentation here!
def runTests(cb:CaseBase, numIterations:int, printResults:bool = True, partialFeatureValidationMax:int = None):
    if partialFeatureValidationMax is not None:
        results = {}
        for j in range(1, partialFeatureValidationMax+1):
                results[j] = []
        for k in range(numIterations):
            for i in range(1, partialFeatureValidationMax+1):
                results[i].append(partialFeatureValidation(cb, 1000, i))
            if printResults:
                print("finished iteration", k) #might be unnecessary
        if printResults:
            for l in range(numIterations):
                printString = str(l)
                for key in results.keys():
                    printString += "," + str(results[key][l])
                print(printString)
            average = "average"
            std = "stdev"
        resultStats = {}
        for key in results.keys():
            resultStats[key] = (sum(results[key]) / numIterations, statistics.stdev(results[key]))
            if printResults:
                average += "," + str(sum(results[key]) / numIterations)
                std += "," + str(statistics.stdev(results[key]))
        if printResults:
            print(average)
            print(std)
        return results
    else: #removal test
        results = []
        for k in range(30):
            results.append(duplicatedFeatureValidation(cb, 1000))
            if printResults:
                print(str(k) + "," + str(results[k]))
        if printResults:
            print(sum(results)/len(results))
            print(statistics.stdev(results))
        return results

#TODO: documentation here!
def runTests_retrain(numIterations:int, features:int, examplesPerAnimal:int, images:list, rootDir:str, useExpertFeatures:str,
                        printResults:bool = True, partialFeatureValidationMax:int = None):
    if partialFeatureValidationMax is not None:
        results = {}
        for j in range(1, partialFeatureValidationMax+1):
                results[j] = []
        for k in range(numIterations):
            tf.keras.backend.clear_session()
            network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
            resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
            extractor = tf.keras.Model(inputs=network.model.input,\
                                        outputs=network.model.layers[len(network.model.layers)-2].output)
            outputs = extractor.predict(resized_images)
            testCB = CaseBase()
            if useExpertFeatures == '0':
                cases = generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, False)
            else:
                cases = generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir)
            for case in cases:
                testCB.addCase(case)
            for i in range(1, partialFeatureValidationMax+1):
                results[i].append(partialFeatureValidation(testCB, 1000, i))
            if printResults:
                print("finished iteration", k) #might be unnecessary
        if printResults:
            for l in range(numIterations):
                printString = str(l)
                for key in results.keys():
                    printString += "," + str(results[key][l])
                print(printString)
            average = "average"
            std = "stdev"
        resultStats = {}
        for key in results.keys():
            resultStats[key] = (sum(results[key]) / numIterations, statistics.stdev(results[key]))
            if printResults:
                average += "," + str(sum(results[key]) / numIterations)
                std += "," + str(statistics.stdev(results[key]))
        if printResults:
            print(average)
            print(std)
        return results
    else: #removal test
        results = []
        for k in range(numIterations):
            tf.keras.backend.clear_session()
            network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
            resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
            extractor = tf.keras.Model(inputs=network.model.input,\
                                        outputs=network.model.layers[len(network.model.layers)-2].output)
            outputs = extractor.predict(resized_images)
            testCB = CaseBase()
            if useExpertFeatures == '0':
                cases = generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, False, False)
            else:
                cases = generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, True, False)
            for case in cases:
                testCB.addCase(case)
            if testCB.caseBaseSize != 50 * examplesPerAnimal:
                print("Race condition error")
                continue
            results.append(duplicatedFeatureValidation(testCB, 1000))
            if printResults:
                print(str(k) + "," + str(results[k]))
        if printResults:
            print(sum(results)/len(results))
            print(statistics.stdev(results))
        return results