import os
import sys
import random
import statistics
import numpy as np
import tensorflow as tf

from case_base import CaseBase
from reader import Reader
from case import Case
from feature import Feature
from deepNetwork import DeepImageNetwork, FeatureNetwork
import helpers

def run(runningSystem:str):
    cb = CaseBase()

    if runningSystem == "local":
        rootDir = "../../expRetrieval_CBR_data/"
    else: #remote
        rootDir = "/l/vision/magnemite/expRetrieval_CBR_data/" #***Need to command "conda activate tensorflow" before running this mode
    numIterations = 30
    partialFeatureValidationMax = 20
    maxNumEpochs = 100

    print("Ready for command:")

    while True:
        userInput = input().split(" ")
        #===================================
        #Import data from file into the case base
        if userInput[0] == "import":
            if len(userInput) == 3:
                for case in Reader().readCSVCases(userInput[1], userInput[2].strip().split(",")):
                    cb.addCase(case)
                print("Import was successful")
            elif len(userInput) == 2:
                for case in Reader().readCSVCases(userInput[1]):
                    cb.addCase(case)
                print("Import was successful")
            elif len(userInput) == 1:
                for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-binary.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
                    cb.addCase(case)
                print("Import was successful")
            else:
                print("Error: incorrect number of arguments")
        #===================================
        #Run tests, considering learned features/weights
        elif userInput[0] == "matchTest":
            # print("Running control...")
            # initialCB = CaseBase()
            # for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-continuous.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
            #     initialCB.addCase(case)
            # helpers.runTests(initialCB, numIterations, True, partialFeatureValidationMax)
            for examplesPerAnimal in [1,5]:
                images = []
                if userInput[4] == "0":
                    images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
                for features in range(10, int(userInput[1])+1, 10):
                    print("==================")
                    print(str(examplesPerAnimal) + " images used per class")
                    print(str(features) + " used in the neural network")
                    if userInput[3] != 'retrain':
                        if userInput[4] == "1":
                            images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
                        invalidImageExistsFlag = True
                        while invalidImageExistsFlag:
                            tf.keras.backend.clear_session()
                            try:
                                network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
                                resized_images = network.train(np.array(images), np.array(labels), 5)
                                invalidImageExistsFlag = False
                            except:
                                print("invalid image found - resetting seed")
                                images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
                                continue
                        extractor = tf.keras.Model(inputs=network.model.input,\
                                                    outputs=network.model.layers[len(network.model.layers)-2].output)
                        outputs = extractor.predict(resized_images)
                        testCB = CaseBase()
                        if userInput[2] == '0':
                            cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, False)
                        else:
                            cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir)
                        for case in cases:
                            testCB.addCase(case)
                        results = helpers.runTests(testCB, numIterations, True, partialFeatureValidationMax)
                    else:
                        results = helpers.runTests_retrain(numIterations, features, examplesPerAnimal, images, rootDir, userInput[2], userInput[4], True, partialFeatureValidationMax)
                    #do anything with reuslts?

        elif userInput[0] == "epochs":
            _, train, classes = Reader().readAwAForNN()
            results = {}
            for k in range(1, maxNumEpochs+1):
                results[k] = []
            for _ in range(numIterations):
                for i in range(1, maxNumEpochs+1):
                    network = FeatureNetwork(None, 85, 50)
                    network.train(np.array(train), np.array(list(classes.values())), i)
                    prediction = network.predict(np.array(train))
                    accuracyCount = 0
                    for j in range(50):
                        if j == np.argmax(prediction[j]):
                            accuracyCount += 1
                    print(i, accuracyCount / 50.0)
                    results[i].append(accuracyCount / 50.0)
            for k in range(1, maxNumEpochs+1):
                print(str(k) + "," + str(sum(results[k]) / float(len(results[k]))))

        elif userInput[0] == "removalTest":
            images = []
            for examplesPerAnimal in [5]: #WARNING - DO NOT use 1! This does not work
                if userInput[4] == "0":
                    images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
                for features in range(10, int(userInput[1])+1, 10):
                    print("==================")
                    print(str(examplesPerAnimal) + " images used per class")
                    print(str(features) + " used in the neural network")
                    if userInput[3] != "retrain":
                        if userInput[4] == "1":
                            images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
                        invalidImageExistsFlag = True
                        while invalidImageExistsFlag:
                            tf.keras.backend.clear_session()
                            try:
                                network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
                                resized_images = network.train(np.array(images), np.array(labels), 5)
                                invalidImageExistsFlag = False
                            except:
                                print("invalid image found - resetting seed")
                                images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
                                continue
                        extractor = tf.keras.Model(inputs=network.model.input,\
                                                    outputs=network.model.layers[len(network.model.layers)-2].output)
                        outputs = extractor.predict(resized_images)
                        testCB = CaseBase()
                        if userInput[2] == '0':
                            cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, False, False)
                        else:
                            cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, True, False)
                        for case in cases:
                            testCB.addCase(case)
                        if testCB.caseBaseSize != 50 * examplesPerAnimal:
                            print("Race condition error")
                            continue
                        else:
                            results = helpers.runTests(testCB, numIterations)
                    else:
                        helpers.runTests_retrain(numIterations, features, examplesPerAnimal, images, rootDir, userInput[2], userInput[4])
        
        elif userInput[0] == "weightTest":
            predicates, train, classes = Reader().readAwAForNN()
        #         results = {}
        #         for j in range(1, 101):
        #             results[j/100.0] = []
        #         for k in range(30):
        #             for i in range(1, 41):
        #                 testCB = CaseBase()
        #                 # for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-binary.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                 for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-continuous.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                     testCB.addCase(case)
        #                 # print("before", testCB.cases[list(testCB.cases.keys())[0]].features)
        #                 initial = datasetTests.partialFeatureValidation(testCB, 1000, i/100.0)
        #                 network = FeatureNetwork(None, 85, 50)
        #                 network.train(np.array(train), np.array(list(classes.values())), 80)
        #                 if i % 10 > 7:
        #                     print(i, network.model.trainable_weights[0].numpy()[0]) #RELU as proper activation function?
        #                 for case in testCB.cases.keys():
        #                     absoluteMax = 0.0
        #                     for feature in testCB.cases[case].features.keys():
        #                         newWeights = network.model.trainable_weights[0].numpy()[predicates[feature]]
        #                         weight = abs(newWeights[classes[testCB.cases[case].result[0]]])
        #                         if weight > absoluteMax:
        #                             absoluteMax = weight
        #                         testCB.cases[case].features[feature].setWeight(weight) #TODO: account for regression
        #                     for feature in testCB.cases[case].features.keys():
        #                         featureObject = testCB.cases[case].features[feature]
        #                         featureObject.setWeight(featureObject.getWeight() / absoluteMax)
        #                 # print("after", testCB.cases[list(testCB.cases.keys())[0]].features)
        #                 final = datasetTests.partialFeatureValidation(testCB, 1000, i/100.0)
        #                 results[i/100.0].append((initial, final))
        #                 print(i)
        #             print("finished iteration", k)
        #         for key in results.keys():
        #             avgInit = 0.0
        #             avgFinal = 0.0
        #             for init, final in results[key]:
        #                 avgInit += init
        #                 avgFinal += final
        #             if len(results[key]) != 0:
        #                 print(str(key) + "," + str(avgInit / len(results[key])) + "," + str(avgFinal / len(results[key])))


        #===================================
        #exit the program
        elif userInput[0] == "q":
            print("Terminating program")
            break
        #===================================
        else:
            print("Error: unknown command")

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: incorrect number of arguments: " + str(len(sys.argv))))

    run(sys.argv[1])