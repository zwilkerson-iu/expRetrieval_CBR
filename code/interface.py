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
            for examplesPerAnimal in [5]:
                images = []
                if userInput[4] == "0":
                    images = helpers.generateImageSample(examplesPerAnimal, rootDir)
                for features in range(10, int(userInput[1])+1, 10):
                    print("==================")
                    print(str(examplesPerAnimal) + " images used per class")
                    print(str(features) + " used in the neural network")
                    if userInput[3] != 'retrain':
                        if userInput[4] == "1":
                            images = helpers.generateImageSample(examplesPerAnimal, rootDir)
                        invalidImageExistsFlag = True
                        while invalidImageExistsFlag:
                            tf.keras.backend.clear_session()
                            try:
                                network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
                                resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
                                invalidImageExistsFlag = False
                            except:
                                print("invalid image found - resetting seed")
                                images = helpers.generateImageSample(examplesPerAnimal, rootDir)
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
                    images = helpers.generateImageSample(examplesPerAnimal, rootDir)
                for features in range(10, int(userInput[1])+1, 10):
                    print("==================")
                    print(str(examplesPerAnimal) + " images used per class")
                    print(str(features) + " used in the neural network")
                    if userInput[3] != "retrain":
                        if userInput[4] == "1":
                            images = helpers.generateImageSample(examplesPerAnimal, rootDir)
                        invalidImageExistsFlag = True
                        while invalidImageExistsFlag:
                            tf.keras.backend.clear_session()
                            try:
                                network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
                                resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
                                invalidImageExistsFlag = False
                            except:
                                print("invalid image found - resetting seed")
                                images = helpers.generateImageSample(examplesPerAnimal, rootDir)
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

        #===================================
        #Query the case base using parameters that are assembled into a query case
        # elif userInput[0] == "query":
        #     if len(userInput) == 2:
        #         args = userInput[1].split(",")
        #         queryCase = Case({}, "")
        #         for i in range(len(args)):
        #             if len(args[i].split(":")) != 2:
        #                 print("Error: invalid syntax")
        #                 return
        #             name, value = args[i].split(":")
        #             try:
        #                 queryCase.addNewFeature(Feature(name, float(value), 1, "euclideanDistance"))
        #             except ValueError:
        #                 queryCase.addNewFeature(Feature(name, value, 1, "match"))
        #         print("Query case: ", queryCase)
        #         result = cb.getClosestCase(queryCase)
        #         print("Closest case: ", result[0], "\nDistance: ", result[1])
        #     else:
        #         print("Error: incorrect number of arguments")
        # #===================================
        # elif userInput[0] == "nn":

        #     if userInput[1] == "imageTest":
        #         examples = int(userInput[2])
        #         for examplesPerAnimal in range(1, examples+1):
        #             for features in range(160, int(userInput[3])+1, 20):
        #                 print("==================")
        #                 print(str(examplesPerAnimal) + " images used per class (non randomized)")
        #                 print(str(features) + " used in the neural network")
        #                 images = []
        #                 classes = os.listdir("data/awa2/JPEGImages")
        #                 for animal in classes:
        #                     imageFiles = os.listdir("data/awa2/JPEGImages/" + animal)
        #                     imageTemps = random.sample(imageFiles, examplesPerAnimal)
        #                     for filepath in imageTemps:
        #                         temp = imread("data/awa2/JPEGImages/" + animal + "/" + filepath, as_gray = False)
        #                         images.append(temp)
        #                 print("images found")
        #                 network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
        #                 print("network created")
        #                 resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
        #                 print("finished training")

        #                 #extraction based on the following sites:
        #                 #   https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
        #                 #   https://stackoverflow.com/questions/47360148/how-to-get-the-output-of-dense-layer-as-a-numpy-array-using-keras-and-tensorflow
        #                 #   https://www.tensorflow.org/api_docs/python/tf/keras/Model?version=nightly
        #                 #   https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer?version=nightly
        #                 extractor = tf.keras.Model(inputs=network.model.input,\
        #                                             outputs=network.model.layers[len(network.model.layers)-2].output)
        #                 outputs = extractor.predict(resized_images)
        #                 # print(outputs[0])
        #                 initialCB = CaseBase()
        #                 # for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-binary.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                 for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-continuous.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                     initialCB.addCase(case)
        #                 testCB = CaseBase()
        #                 # for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-binary.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                 for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-continuous.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                     index = classes.index(case.result[0])
        #                     for i in range(len(outputs[0])):
        #                         sumTemp = 0.0
        #                         for j in range(examplesPerAnimal):
        #                             sumTemp += outputs[index * examplesPerAnimal + j][i]
        #                         case.addNewFeature(Feature("Feature" + str(i), sumTemp / examplesPerAnimal, 1, "euclideanDistance"))
        #                     testCB.addCase(case)
        #                 print("case bases set up without errors")

        #                 results = {}
        #                 for j in range(1, 20, 1):
        #                      results[j] = []
        #                 # for j in range(1, 15, 1):
        #                 #     results[j/100.0] = []
        #                 for k in range(30):
        #                     for i in range(1, 20, 1):
        #                     # for i in range(1, 15, 1): #TODO: do we need to retrain the neural network too?
        #                         # results[i/100.0].append((datasetTests.partialFeatureValidation(initialCB, 1000, (features + 85)/85.0 * i/100.0),\
        #                         #                             datasetTests.partialFeatureValidation(testCB, 1000, i/100.0)))
        #                         results[i].append((datasetTests.partialFeatureValidation(initialCB, 1000, i),\
        #                                                     datasetTests.partialFeatureValidation(testCB, 1000, i)))
        #                         # print(i)
        #                     print("finished iteration", k)
        #                 for key in results.keys():
        #                     avgInit = 0.0
        #                     avgFinal = 0.0
        #                     for init, final in results[key]:
        #                         avgInit += init
        #                         avgFinal += final
        #                     if len(results[key]) != 0:
        #                         print(str(key) + "," + str(avgInit / len(results[key])) + "," + str(avgFinal / len(results[key])))
            
        #     elif userInput[1] == "epochs":
        #         predicates, train, classes = Reader().readAwAForNN()
        #         results = {}
        #         for k in range(1, 101):
        #             results[k] = []
        #         for _ in range(30):
        #             for i in range(1, 101):
        #                 network = FeatureNetwork(None, 85, 50)
        #                 network.train(np.array(train), np.array(list(classes.values())), i)
        #                 prediction = network.predict(np.array(train))
        #                 accuracyCount = 0
        #                 for j in range(50):
        #                     if j == np.argmax(prediction[j]):
        #                         accuracyCount += 1
        #                 print(i, accuracyCount / 50.0)
        #                 results[i].append(accuracyCount / 50.0)
        #         for k in range(1, 101):
        #             print(str(k) + "," + str(sum(results[k]) / float(len(results[k]))))

        #     elif userInput[1] == "weightTest":
        #         predicates, train, classes = Reader().readAwAForNN()
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
        
        # #===================================
        # elif userInput[0] == "diverseCB":
        #     for examplesPerAnimal in [5,10]: #WARNING - DO NOT use 1! This does not work
        #         for features in range(20, int(userInput[1])+1, 20):
        #             print("==================")
        #             print(str(examplesPerAnimal) + " images used per class")
        #             print(str(features) + " learned from the neural network")
        #             images = []
        #             classes = os.listdir("data/awa2/JPEGImages")
        #             for animal in classes:
        #                 imageFiles = os.listdir("data/awa2/JPEGImages/" + animal)
        #                 imageTemps = random.sample(imageFiles, examplesPerAnimal)
        #                 for filepath in imageTemps:
        #                     temp = imread("data/awa2/JPEGImages/" + animal + "/" + filepath, as_gray = False)
        #                     images.append(temp)
        #             print("images found")
        #             network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
        #             print("network created")
        #             resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
        #             print("finished training")

        #             extractor = tf.keras.Model(inputs=network.model.input,\
        #                                         outputs=network.model.layers[len(network.model.layers)-2].output)
        #             outputs = extractor.predict(resized_images)

        #             # initialCB = CaseBase()
        #             # for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-continuous.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #             #     initialCB.addCase(case)
        #             testCB = CaseBase()
        #             caseList = []
        #             counter = 0
        #             for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-continuous.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                 index = classes.index(case.result[0])
        #                 for j in range(examplesPerAnimal):
        #                     temp = Case({}, (case.result[0], 1.0))
        #                     for feature in case.features.values():
        #                         temp.addNewFeature(feature)
        #                     for i in range(len(outputs[0])):
        #                         temp.addNewFeature(Feature("Feature" + str(i), float(outputs[index * examplesPerAnimal + j][i]), 1, "euclideanDistance"))
        #                     caseList.append(temp)
        #             for newCase in caseList:
        #                 testCB.addCase(newCase)
        #                 counter += 1
        #             print("case bases set up without errors") #TODO: still seem to be encountering a race condition... while loop?
        #             print("testCB size:", testCB.caseBaseSize, counter, len(list(testCB.cases.keys())))
        #             if testCB.caseBaseSize != counter:
        #                 print("ERROR: retest required")
        #                 continue

        #             # print(testCB.maxima)
        #             results = []
        #             for k in range(30):
        #                 # results.append((datasetTests.duplicatedFeatureValidation(initialCB, 1000),\
        #                 #                     datasetTests.duplicatedFeatureValidation(testCB, 1000)))
        #                 results.append(datasetTests.duplicatedFeatureValidation(testCB, 1000))
        #                 print(str(k) + "," + str(results[k]))
        #             print(sum(results)/len(results), statistics.stdev(results))
        #             # avgInit = 0.0
        #             # avgFinal = 0.0
        #             # for init, final in results:
        #             #     avgInit += init
        #             #     avgFinal += final
        #             # print(str(avgInit / len(results)) + "," + str(avgFinal / len(results)))

        # elif userInput[0] == "onlyNN":
        #     for examplesPerAnimal in ([10]):
        #         for features in range(100, int(userInput[1])+1, 20):
        #             print("==================")
        #             print(str(examplesPerAnimal) + " images used per class")
        #             print(str(features) + " learned from the neural network")
        #             images = []
        #             classes = os.listdir("data/awa2/JPEGImages")
        #             for animal in classes:
        #                 imageFiles = os.listdir("data/awa2/JPEGImages/" + animal)
        #                 imageTemps = random.sample(imageFiles, examplesPerAnimal)
        #                 for filepath in imageTemps:
        #                     temp = imread("data/awa2/JPEGImages/" + animal + "/" + filepath, as_gray = False)
        #                     images.append(temp)
        #             print("images found")
        #             network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
        #             print("network created")
        #             resized_images = network.train(np.array(images), np.array([0] * len(images)), 5)
        #             print("finished training")

        #             extractor = tf.keras.Model(inputs=network.model.input,\
        #                                         outputs=network.model.layers[len(network.model.layers)-2].output)
        #             outputs = extractor.predict(resized_images)

        #             initialCB = CaseBase()
        #             for case in Reader().readAwADataFromTxt("data/awa2/predicate-matrix-continuous.txt", "data/awa2/classes.txt", "data/awa2/predicates.txt"):
        #                 initialCB.addCase(case)
        #             testCB = CaseBase()
        #             counter = 0
        #             caseList = []
        #             for caseName in classes:
        #                 temp = Case({}, (caseName, 1.0))
        #                 index = classes.index(caseName)
        #                 for i in range(len(outputs[0])):
        #                     sumTemp = 0.0
        #                     for j in range(examplesPerAnimal):
        #                         sumTemp += outputs[index * examplesPerAnimal + j][i]
        #                     temp.addNewFeature(Feature("Feature" + str(i), sumTemp / examplesPerAnimal, 1, "euclideanDistance"))
        #                 caseList.append(temp)
        #                 counter += 1
        #             for newCase in caseList:
        #                 testCB.addCase(newCase)
        #             print("case bases set up without errors") #TODO: still seem to be encountering a race condition... while loop?
        #             print("testCB size:", testCB.caseBaseSize, counter, len(list(testCB.cases.keys())))

        #             results = {}
        #             for j in range(1, 20, 1):
        #                     results[j] = []
        #             # for j in range(1, 15, 1):
        #             #     results[j/100.0] = []
        #             for k in range(30):
        #                 for i in range(1, 20, 1):
        #                 # for i in range(1, 15, 1): #TODO: do we need to retrain the neural network too?
        #                     # results[i/100.0].append((datasetTests.partialFeatureValidation(initialCB, 1000, (features + 85)/85.0 * i/100.0),\
        #                     #                             datasetTests.partialFeatureValidation(testCB, 1000, i/100.0)))
        #                     results[i].append((datasetTests.partialFeatureValidation(initialCB, 1000, i),\
        #                                                 datasetTests.partialFeatureValidation(testCB, 1000, i)))
        #                     # print(i)
        #                 print("finished iteration", k)
        #             for key in results.keys():
        #                 avgInit = 0.0
        #                 avgFinal = 0.0
        #                 for init, final in results[key]:
        #                     avgInit += init
        #                     avgFinal += final
        #                 if len(results[key]) != 0:
        #                     print(str(key) + "," + str(avgInit / len(results[key])) + "," + str(avgFinal / len(results[key])))