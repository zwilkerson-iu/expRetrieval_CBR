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
    if runningSystem == "local":
        rootDir = "../../expRetrieval_CBR_data/"
    else: #remote
        rootDir = "/l/vision/magnemite/expRetrieval_CBR_data/" #***Need to command "conda activate tensorflow" before running this mode
    NUMITERATIONS = 30

    print("Ready for command:")

    while True:
        userInput = input().split(" ")
        
        #===================================
        #Run tests, considering learned features/weights
        
        # UserInput key:
        # 0 = test key [0 = expert, 1 = learned, 2 = mixed]
        # 1 = randomness bound [1,10]
        # 2 = weights used key [0 = False, 1 = True] TODO: implement this
        if int(userInput[0]) <= 2:
            for examplesPerAnimal in [10, 20, 50, 100]:
                if int(userInput[1]) != 0:
                    for randomness in range(1, int(userInput[1])+1):
                        for features in [512]:
                            print("==================")
                            print(str(examplesPerAnimal) + " cases used per class")
                            print(str(features) + " features used in neural network dense layers")
                            if int(userInput[0]) == 0 or int(userInput[0]) == 2:
                                helpers.runTests(NUMITERATIONS, features, examplesPerAnimal, rootDir, int(userInput[0]), randomness, int(userInput[2]))
                            else:
                                helpers.runTests(NUMITERATIONS, features, examplesPerAnimal, rootDir, int(userInput[0]), 0, int(userInput[2]))
                else:
                    for features in [512]:
                        print("==================")
                        print(str(examplesPerAnimal) + " cases used per class")
                        print(str(features) + " features used in neural network dense layers")
                        if int(userInput[0]) == 0 or int(userInput[0]) == 2:
                            helpers.runTests(NUMITERATIONS, features, examplesPerAnimal, rootDir, int(userInput[0]), randomness, int(userInput[2]))
                        else:
                            helpers.runTests(NUMITERATIONS, features, examplesPerAnimal, rootDir, int(userInput[0]), 0, int(userInput[2]))                    

        # UserInput key:
        # 0 = test key [3 = epochs]
        # 1 = modal key [0 = expert (weights), 1 = learned (features), 2 = mixed (weights)]
        # 2 = maximum number of epochs tested [1,100]
        elif int(userInput[0]) == 3:
            maxNumEpochs = int(userInput[2])
            results = {}
            if int(userInput[1]) == 0:
                for k in range(1, maxNumEpochs+1):
                    results[k] = []
                _, train, classes = Reader().readAwAForNN()
                for _ in range(NUMITERATIONS):
                    for i in range(1, maxNumEpochs+1):
                        tf.keras.backend.clear_session()
                        network = FeatureNetwork(None, 85, 50)
                        network.train(np.array(train), np.array(list(classes.values())), i)
                        prediction = network.predict(np.array(train))
                        accuracyCount = 0
                        for j in range(50):
                            if j == np.argmax(prediction[j]):
                                accuracyCount += 1
                        results[i].append(accuracyCount / 50.0)
                for k in range(1, maxNumEpochs+1):
                    print(str(k) + "," + str(sum(results[k]) / float(len(results[k]))))
            elif int(userInput[1]) == 1:
                for k in range(1, maxNumEpochs+1):
                    results[k] = ([],[])
                for _ in range(NUMITERATIONS):
                    for i in range(1, maxNumEpochs+1):
                        images, labels = helpers.generateImageSample(40, rootDir, 0, weightsUsed=maxNumEpochs)
                        train_images, train_labels = images[:20*50], labels[:20*50]
                        test_images, test_labels = images[20*50:], labels[20*50:]
                        print(np.array(train_images).shape)
                        print(len(train_images))
                        invalidImageExistsFlag = True
                        while invalidImageExistsFlag:
                            try:
                                tf.keras.backend.clear_session()
                                network = DeepImageNetwork(numFeatures=4096)
                                resized_train = network.train(np.array(train_images), np.array(train_labels), numEpochs=i)
                                resized_test = np.empty((len(test_images), 227, 227, 3))
                                for q in range(len(test_images)):
                                    resized_test[q] = tf.image.resize(tf.image.per_image_standardization(test_images[q]), (227,227))
                                invalidImageExistsFlag = False
                            except:
                                print("invalid image found - resetting seed")
                                images, labels = helpers.generateImageSample(40, rootDir, 0, weightsUsed=maxNumEpochs)
                                train_images, train_labels = images[:20*50], labels[:20*50]
                                test_images, test_labels = images[20*50:], labels[20*50:]
                                continue
                        train_pred = network.predict(resized_train)
                        test_pred = network.predict(resized_test)
                        accuracyCount = 0
                        for j in range(len(train_labels)):
                            if train_labels[j] == np.argmax(train_pred[j]):
                                accuracyCount += 1
                        results[i][0].append(accuracyCount / 50.0)
                        accuracyCount = 0
                        for j in range(len(test_labels)):
                            if test_labels[j] == np.argmax(test_pred[j]):
                                accuracyCount += 1
                        results[i][1].append(accuracyCount / 50.0)
                for k in range(1, maxNumEpochs+1):
                    print(str(k) + "," + str(sum(results[k][0]) / float(len(results[k][0]))) + "," + str(sum(results[k][1]) / float(len(results[k][1]))))
            elif int(userInput[1]) == 2:
                pass
            #TODO: implement
            if userInput[1] == 0:
                record = open("../results/" + userInput[0] + "_" + userInput[1] + "_" + userInput[2] + "_" + "_results.csv", "w")
                for iteration in results.keys():
                    record.write(str(iteration) + "," + ",".join(results[iteration]) + "\n")
                record.close()
            else:
                record = open("../results/" + userInput[0] + "_" + userInput[1] + "_" + userInput[2] + "_" + "_results.csv", "w")
                for iteration in results.keys():
                    record.write(str(iteration) + "," + ",".join(results[iteration][0]) + "\n")
                    record.write(str(iteration) + "," + ",".join(results[iteration][1]) + "\n")
                record.close()

        
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
        #Import data from file into the case base
        # if userInput[0] == "import":
        #     if len(userInput) == 3:
        #         for case in Reader().readCSVCases(userInput[1], userInput[2].strip().split(",")):
        #             cb.addCase(case)
        #         print("Import was successful")
        #     elif len(userInput) == 2:
        #         for case in Reader().readCSVCases(userInput[1]):
        #             cb.addCase(case)
        #         print("Import was successful")
        #     elif len(userInput) == 1:
        #         for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-binary.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
        #             cb.addCase(case)
        #         print("Import was successful")
        #     else:
        #         print("Error: incorrect number of arguments")

# if userInput[0] == "matchTest":
#             # print("Running control...")
#             # initialCB = CaseBase()
#             # for case in Reader().readAwADataFromTxt(rootDir + "awa2/predicate-matrix-continuous.txt", rootDir + "awa2/classes.txt", rootDir + "awa2/predicates.txt"):
#             #     initialCB.addCase(case)
#             # helpers.runTests(initialCB, numIterations, True, partialFeatureValidationMax)
#             for examplesPerAnimal in [1,5]:
#                 images = []
#                 if userInput[4] == "0":
#                     images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
#                 for features in range(10, int(userInput[1])+1, 10):
#                     print("==================")
#                     print(str(examplesPerAnimal) + " images used per class")
#                     print(str(features) + " used in the neural network")
#                     if userInput[3] != 'retrain':
#                         if userInput[4] == "1":
#                             images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
#                         invalidImageExistsFlag = True
#                         while invalidImageExistsFlag:
#                             tf.keras.backend.clear_session()
#                             try:
#                                 network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
#                                 resized_images = network.train(np.array(images), np.array(labels), 5)
#                                 invalidImageExistsFlag = False
#                             except:
#                                 print("invalid image found - resetting seed")
#                                 images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
#                                 continue
#                         extractor = tf.keras.Model(inputs=network.model.input,\
#                                                     outputs=network.model.layers[len(network.model.layers)-2].output)
#                         outputs = extractor.predict(resized_images)
#                         testCB = CaseBase()
#                         if userInput[2] == '0':
#                             cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, False)
#                         else:
#                             cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir)
#                         for case in cases:
#                             testCB.addCase(case)
#                         results = helpers.runTests(testCB, numIterations, True, partialFeatureValidationMax)
#                     else:
#                         results = helpers.runTests_retrain(numIterations, features, examplesPerAnimal, images, rootDir, userInput[2], userInput[4], True, partialFeatureValidationMax)
#                     #do anything with reuslts?

# elif userInput[0] == "removalTest":
#             images = []
#             for examplesPerAnimal in [5]: #WARNING - DO NOT use 1! This does not work
#                 if userInput[4] == "0":
#                     images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
#                 for features in range(10, int(userInput[1])+1, 10):
#                     print("==================")
#                     print(str(examplesPerAnimal) + " images used per class")
#                     print(str(features) + " used in the neural network")
#                     if userInput[3] != "retrain":
#                         if userInput[4] == "1":
#                             images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
#                         invalidImageExistsFlag = True
#                         while invalidImageExistsFlag:
#                             tf.keras.backend.clear_session()
#                             try:
#                                 network = DeepImageNetwork(None, (1200, 1200), 50, numFeatures=features)
#                                 resized_images = network.train(np.array(images), np.array(labels), 5)
#                                 invalidImageExistsFlag = False
#                             except:
#                                 print("invalid image found - resetting seed")
#                                 images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir)
#                                 continue
#                         extractor = tf.keras.Model(inputs=network.model.input,\
#                                                     outputs=network.model.layers[len(network.model.layers)-2].output)
#                         outputs = extractor.predict(resized_images)
#                         testCB = CaseBase()
#                         if userInput[2] == '0':
#                             cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, False, False)
#                         else:
#                             cases = helpers.generateCaseListWithLearnedFeatures(outputs, examplesPerAnimal, rootDir, True, False)
#                         for case in cases:
#                             testCB.addCase(case)
#                         if testCB.caseBaseSize != 50 * examplesPerAnimal:
#                             print("Race condition error")
#                             continue
#                         else:
#                             results = helpers.runTests(testCB, numIterations)
#                     else:
#                         helpers.runTests_retrain(numIterations, features, examplesPerAnimal, images, rootDir, userInput[2], userInput[4])