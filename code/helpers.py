from case_base import CaseBase
from case import Case
from reader import Reader
from feature import Feature
from deepNetwork import DeepImageNetwork, FeatureNetwork

from skimage.io import imread
import numpy as np
import tensorflow as tf
import os
import sys
import random
import statistics

"""
Computes KNN majority value for a given set of cases (currently applied only to 1NN scenarios)
- caseBase = the case base from which neighbor cases are drawn
- neighbors = the list of closest neighbors to some query case
- mode = (currently unused) means of customizing the definition (e.g., vote-based vs. distance-based)
Returns: the majority value
"""
def majorityRule(caseBase:CaseBase, neighbors:list, mode:str = "majority"):
    if type(caseBase.retrieveCase(neighbors[0][0]).result) is tuple:
        regressionFlag = False
    else:
        regressionFlag = True
    resultOptions = {}
    if mode == "majority":
        for caseID, _, _ in neighbors:
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

    elif mode == "weighted": #TODO: implement weighted voting/average...?
        print("TODO: implement this")
    else:
        print("Error: invalid mode parameter")

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
    for i in range(numberOfIterations):
        # caseIndex = random.randint(0, caseBase.caseBaseSize - 1)
        # case = caseBase.retrieveCase(list(caseBase.cases.keys())[caseIndex])
        case = caseBase.retrieveCase(list(caseBase.cases.keys())[i])
        neighbors = caseBase.getKClosestCases(case, k)
        queryResult = majorityRule(caseBase, neighbors[1:])
        if queryResult == case.result:
            correctClassifications += 1
    return correctClassifications / float(numberOfIterations)

"""
Generates random samples of images pulling equally from all classes for CNN training (and testing) purposes
- numImagesPerAnimal = the number of samples per class to select
- rootDir = the root directory key defining where image files are located (depending on whether testing is done locally or remotely)
- featureSelectionMode = type of test calling this function (for file naming purposes only)
- randomBound = randomness value relative to expert features in calling function (for file naming purposes only)
- weightsUsed = whether dynamic weighting is used in the calling function (for file naming purposes only, can also be used as a placeholder for maximum
                number of epochs used in the calling function)
Returns: A tuple containing numpy arrays of selected images and their corresponding labels
"""
def generateImageSample(numImagesPerAnimal:int, rootDir:str, k:int, featureSelectionMode:int = -1, randomBound:int = 0, weightsUsed:int = 0):
    imageRecord = {}
    images = []
    labels = np.empty(50 * numImagesPerAnimal)
    classes = os.listdir(rootDir + "awa2/JPEGImages")
    for index in range(len(classes)):
        animal = classes[index]
        imageFiles = os.listdir(rootDir + "awa2/JPEGImages/" + animal)
        imageTemps = random.sample(imageFiles, numImagesPerAnimal)
        imageRecord[animal] = imageTemps
        #print(animal + "," + ",".join(x for x in imageTemps))
        for f in range(len(imageTemps)):
            temp = imread(rootDir + "awa2/JPEGImages/" + animal + "/" + imageTemps[f], as_gray = False)
            images.append(temp)
            labels[index*numImagesPerAnimal+f] = index
    record = open("../csvFiles/" + str(featureSelectionMode) + "_" + str(randomBound) + "_" + str(weightsUsed) + "_" + str(k) + ".csv", "w")
    for animal in imageRecord.keys():
        record.write(animal + "," + ",".join(x for x in imageRecord[animal]) + "\n")
    record.close()
    return (images, labels)

"""
Creates the list of cases to be added to the case base in a given test iteration
- numImagesPerAnimal = the number of examples per class being considered
- rootDir = root directory path segment relative to local or remote testing
- featureSelectionMode = type of test calling this function (i.e., expert, learned, or a combination)
- learnedFeatures = where applicable, the set of learned features from the CNN
- randomBound = the randomness applied to the expert features
- featureFraction = when testing the combination of expert and learned features, a tuple representing the percentage of each feature set to consider per-case
                    (will always sum to 100 unless testing 100% of both feature sets)
Returns: A set of case objects to be added to the case base
"""
def generateCaseList(numImagesPerAnimal:int, rootDir:str, featureSelectionMode:int, learnedFeatures = None, randomBound:int = 0, featureFraction:tuple = None):
    casesRet = []
    classes = os.listdir(rootDir + "awa2/JPEGImages")
    if featureSelectionMode == 2:
        cases = Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-continuous.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt")
        expertFeatures = random.sample(tuple(cases[0].features.keys()), int(0.01*featureFraction[0]*len(cases[0].features.keys())))
        nnFeatures = random.sample((range(0, len(learnedFeatures[0]))), int(0.01*featureFraction[1]*len(learnedFeatures[0])))
        for case in cases:
            index = classes.index(case.result[0])
            for j in range(numImagesPerAnimal):
                temp = Case({}, (case.result[0], 1.0))
                for featureName in case.features.keys():
                    if featureName in expertFeatures:
                        temp.addNewFeature(Feature(featureName, case.features[featureName].getValue(), case.features[featureName].getWeight()))
                        if randomBound > 0:
                            r = float(random.randint(1, randomBound)) #NOTE: can make into random.uniform to allow for float randomBound values
                            if random.randint(0, 1) == 0:
                                r = 1.0/r
                            temp.editFeature(featureName, temp.getFeature(featureName).getValue() * r)
                for i in range(len(learnedFeatures[0])):
                    if i in nnFeatures:
                        temp.addNewFeature(Feature("Feature" + str(i), float(learnedFeatures[index * numImagesPerAnimal + j][i]), 1, "euclideanDistance"))
                casesRet.append(temp)
    elif featureSelectionMode == 1:
        for caseName in classes:
            index = classes.index(caseName)
            for j in range(numImagesPerAnimal):
                temp = Case({}, (caseName, 1.0))
                for i in range(len(learnedFeatures[0])):
                    temp.addNewFeature(Feature("Feature" + str(i), float(learnedFeatures[index * numImagesPerAnimal + j][i]), 1, "euclideanDistance"))
                casesRet.append(temp)
    elif featureSelectionMode == 0:
        for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-continuous.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
            for j in range(numImagesPerAnimal):
                temp = Case({}, (case.result[0], 1.0))
                for featureName in case.features.keys():
                    temp.addNewFeature(Feature(featureName, case.features[featureName].getValue(), case.features[featureName].getWeight()))
                    if randomBound > 0:
                        r = float(random.randint(1, randomBound)) #Opt. TODO can make into random.uniform to allow for float randomBound values
                        if random.randint(0, 1) == 0:
                            r = 1.0/r
                        # r = random.uniform(1.0 - randomBound * 0.01, 1.0 + randomBound * 0.01)
                        # print(r)
                        temp.editFeature(featureName, temp.getFeature(featureName).getValue() * r)
                casesRet.append(temp)
    else:
        print("ERROR: invalid feature selection mode")
    return casesRet

"""
Base testing function for the case-based reasoner
- numIterations = a tuple containing the start and end iteration values for an experiment (facilitating parallelism)
- features = the number of features to be applied in dense layers of the CNN, where applicable
- examplesPerAnimal = the number of examples per class considered
- rootDir = root directory path segment relative to local or remote testing
- featureSelectionMode = type of test calling this function (i.e., expert, learned, or a combination)
- randomBound = the randomness applied to the expert features
- weightsUsed = key defining whether learned weights are also tested
Returns: a dictionary organizing testing results
"""
def runTests(numIterations:tuple, features:int, examplesPerAnimal:int, rootDir:str, featureSelectionMode:int, randomBound:int = 0, weightsUsed:int = 0, sigma:int = 80):
    results = {}
    outputs = None
    for k in range(numIterations[0], numIterations[1]):
        if results.get(k) == None:
            results[k] = {}
        if featureSelectionMode == 1 or featureSelectionMode == 2: #All learned, or mixed
            images, labels = generateImageSample(examplesPerAnimal, rootDir, k, featureSelectionMode, randomBound, weightsUsed)
            invalidImageExistsFlag = True
            while invalidImageExistsFlag:
                try:
                    tf.keras.backend.clear_session()
                    network = DeepImageNetwork(numFeatures=features)
                    resized_images = network.train(np.array(images), labels, numEpochs=50)
                    invalidImageExistsFlag = False
                except:
                    print("invalid image found - resetting seed")
                    images, labels = generateImageSample(examplesPerAnimal, rootDir, k, featureSelectionMode, randomBound, weightsUsed)
                    continue
            extractor = tf.keras.Model(inputs=network.model.input,\
                                        outputs=network.model.layers[len(network.model.layers)-2].output)
            outputs = extractor.predict(resized_images)

        caseBases = {}
        if randomBound == 11:
            randStart = 1
            randEnd = randomBound
        else:
            randStart = randomBound
            randEnd = randomBound+1
        for randomness in range(randStart, randEnd):
            results[k][randomness] = []
            caseBases[randomness] = []
            if featureSelectionMode == 2:
                for frac in range(10, 91, 20):
                    featureFrac = (frac, 100-frac)
                    cb = CaseBase()
                    for case in generateCaseList(examplesPerAnimal, rootDir, featureSelectionMode, outputs, randomness, featureFrac):
                        cb.addCase(case)
                    caseBases[randomness].append(cb)
            cb = CaseBase()
            for case in generateCaseList(examplesPerAnimal, rootDir, featureSelectionMode, outputs, randomness, (100,100)):
                cb.addCase(case)
            caseBases[randomness].append(cb)
        print("case bases created")

        for randomness in caseBases.keys():
            for cb in caseBases[randomness]:
                if weightsUsed == 1 or weightsUsed == 2: #2 implies using test 4 to generate weights
                    _, _, classes = Reader().readAwAForNN(rootDir)
                    if weightsUsed == 2: #NOTE: these are absolute rather than local weights...
                        if featureSelectionMode == 1:
                            newWeights = generateWeights(cb, examplesPerAnimal, classes, sigma, 3) # newWeights needs to be a parameter that can only be set manually (from interface.py)
                        else:
                            newWeights = generateWeights(cb, examplesPerAnimal, classes, sigma)
                        for caseHash in cb.cases.keys():
                            for featureName in cb.cases[caseHash].features.keys():
                                cb.cases[caseHash].features[featureName].setWeight(newWeights[featureName])
                        print("new weights used")

                    else:
                        if featureSelectionMode == 0:
                            numFeatures = 85
                        elif featureSelectionMode == 1:
                            numFeatures = 1024
                        else:
                            numFeatures = 1109
                        featureSet = np.empty((cb.caseBaseSize, numFeatures))
                        weightLabels = np.empty(cb.caseBaseSize)
                        keys = tuple(cb.cases.keys())
                        for c in range(len(keys)):
                            weightFeatures = tuple(cb.cases[keys[c]].features.keys())
                            for f in range(len(weightFeatures)):
                                featureSet[c][f] = cb.cases[keys[c]].features[weightFeatures[f]].value
                            weightLabels[c] = classes[cb.cases[keys[c]].result[0]]
                        weighter = FeatureNetwork(None, numFeatures, 50)
                        if featureSelectionMode == 0:
                            weighter.train(featureSet, weightLabels, 80)
                        elif featureSelectionMode == 1:
                            weighter.train(featureSet, weightLabels, 5)
                        else:
                            weighter.train(featureSet, weightLabels, 5) #TODO: set this based on epochs testing
                        for c in range(len(keys)):
                            weightFeatures = tuple(cb.cases[keys[c]].features.keys())
                            absoluteMax = 0.0
                            for f in range(len(weightFeatures)):
                                newWeights = weighter.model.trainable_weights[0].numpy()[f]
                                weight = abs(newWeights[classes[cb.cases[keys[c]].result[0]]])
                                if weight > absoluteMax:
                                    absoluteMax = weight
                                cb.cases[keys[c]].features[weightFeatures[f]].setWeight(weight)
                            for featureName in weightFeatures:
                                cb.cases[keys[c]].features[featureName].setWeight(cb.cases[keys[c]].features[featureName].getWeight() / absoluteMax)

                    # elif featureSelectionMode == 1:
                    #     newWeights = network.model.trainable_weights[-1].numpy()
                    #     for caseHash in cb.cases.keys():
                    #         absoluteMax = 0.0
                    #         for featureName in cb.cases[caseHash].features.keys():
                    #             weightSet = newWeights[int(featureName[7:])]
                    #             weight = abs(weightSet[classes[cb.cases[caseHash].result[0]]])
                    #             if weight > absoluteMax:
                    #                 absoluteMax = weight
                    #             cb.cases[caseHash].features[featureName].setWeight(weight)
                    #         for featureName in cb.cases[caseHash].features.keys():
                    #             cb.cases[caseHash].features[featureName].setWeight(cb.cases[caseHash].features[featureName].getWeight() / absoluteMax)

                    print("weights generated and applied")
                
                # results[k][randomness].append(duplicatedFeatureValidation(cb, 1000))
                results[k][randomness].append(duplicatedFeatureValidation(cb, cb.caseBaseSize))
                print(str(k) + "," + str(randomness) + ",", results[k][randomness])

    for r in results[k].keys():
        record = open("../results/" + str(featureSelectionMode) + "_" + str(r) + "_" + str(weightsUsed) + "_" + str(k) + "_results" + str(examplesPerAnimal) + ".csv", "w")
        for m in results.keys():
            record.write(str(m) + "," + ",".join(map(str, results[m][r])) + "\n")
        record.close()
    return results

#TODO: documentation
def generateWeights(cb:CaseBase, examplesPerAnimal:int, classes, sigma:int, maxNumEpochs:int = 80):
    weights = {}
    featuresList = tuple(cb.cases[tuple(cb.cases.keys())[0]].features.keys())
    numFeatures = len(featuresList)
    inputs_control = np.empty((examplesPerAnimal*50, numFeatures))
    labels = np.empty(examplesPerAnimal*50)
    counter = [0,0]
    for featureName in featuresList:
        weights[featureName] = 0.0
    for caseHash in cb.cases.keys():            
        for featureName in cb.cases[caseHash].features.keys():
            inputs_control[counter[0]][counter[1]] = cb.cases[caseHash].features[featureName].value
            labels[counter[0]] = classes[cb.cases[caseHash].result[0]]
            counter[1] += 1
        counter[0] += 1
        counter[1] = 0
    print(labels)
    print(len(inputs_control[0]))
    
    tf.keras.backend.clear_session()
    network = FeatureNetwork(None, numFeatures, 50)
    network.train(inputs_control, labels, maxNumEpochs)
    control = network.predict(inputs_control)
    for i in range(numFeatures):
        accuracyCounts = [0,0,0]
        temp_plus = np.empty((examplesPerAnimal*50, numFeatures))
        temp_minus = np.empty((examplesPerAnimal*50, numFeatures))
        for a in range(50):
            for e in range(examplesPerAnimal):
                for f in range(numFeatures):
                    if f != i:
                        temp_plus[a*examplesPerAnimal+e][f] = inputs_control[a*examplesPerAnimal+e][f]
                        temp_minus[a*examplesPerAnimal+e][f] = inputs_control[a*examplesPerAnimal+e][f]
                    else:
                        temp_plus[a*examplesPerAnimal+e][f] = inputs_control[a*examplesPerAnimal+e][f] + sigma*0.01*inputs_control[a*examplesPerAnimal+e][f]
                        temp_minus[a*examplesPerAnimal+e][f] = inputs_control[a*examplesPerAnimal+e][f] - sigma*0.01*inputs_control[a*examplesPerAnimal+e][f]
        plus = network.predict(temp_plus)
        minus = network.predict(temp_minus)
        for j in range(len(labels)):
            if labels[j] == np.argmax(control[j]):
                accuracyCounts[0] += 1
            if labels[j] == np.argmax(plus[j]):
                accuracyCounts[1] += 1
            if labels[j] == np.argmax(minus[j]):
                accuracyCounts[2] += 1
        print(accuracyCounts)
        finalValue = (abs(accuracyCounts[0] - accuracyCounts[1])/float(len(labels)) + abs(accuracyCounts[0] - accuracyCounts[2])/float(len(labels)))/2.0
        weights[featuresList[i]] = finalValue
    return weights