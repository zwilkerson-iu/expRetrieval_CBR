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
    saveDir = "../results/"
    NUMITERATIONS = 5

    while True:
        print("Ready for command:")
        userInput = input().split(" ")
        
        #===================================
        #exit the program
        if userInput[0] == "q":
            print("Terminating program")
            break

        #===================================
        #Perform data analysis/reformatting
        elif userInput[0] == "a":
            arg1 = int(userInput[1])
            arg2 = int(userInput[2])
            try:
                arg3 = int(userInput[3])
                Reader().analyzeData(saveDir, arg1, arg2, arg3)
            except:
                Reader().analyzeData(saveDir, arg1, arg2)

        #===================================
        #Run tests, considering learned features/weights
        
        # UserInput key:
        # 0 = test key [0 = expert, 1 = learned, 2 = mixed]
        # 1 = randomness bound [1,10]
        # 2 = weights used key [0 = False, 1 = True, 2 = New Weights Method]
        # 3 = optional value to set limits of iterations (i.e., x to x+5) for parallelism
        elif int(userInput[0]) <= 2:
            for examplesPerAnimal in [10]: #Maybe add 50 later; maximum is 100 images per class, assuming no invalid ones in the smallest class
                for features in [1024]:
                    print("==================")
                    print(str(examplesPerAnimal) + " cases used per class")
                    print(str(features) + " features used in neural network dense layers (where applicable)")
                    try:
                        helpers.runTests((int(userInput[3]), int(userInput[3])+NUMITERATIONS), features, examplesPerAnimal, rootDir, int(userInput[0]), int(userInput[1]), int(userInput[2]))
                    except:
                        helpers.runTests((0, 30), features, examplesPerAnimal, rootDir, int(userInput[0]), int(userInput[1]), int(userInput[2]))         

        # UserInput key:
        # 0 = test key [3 = epochs]
        # 1 = modal key [0 = expert (weights), 1 = learned (features), 2 = mixed (weights)]
        # 2 = maximum number of epochs tested [1,100]
        # 3 = optional value to set limits of iterations (i.e., x to x+5) for parallelism
        elif int(userInput[0]) == 3:
            try:
                iterStart = int(userInput[3])
            except:
                iterStart = 0
            maxNumEpochs = int(userInput[2])
            results = {}
            if int(userInput[1]) == 0:
                for k in range(1, maxNumEpochs+1):
                    results[k] = []
                _, train, classes = Reader().readAwAForNN(rootDir)
                for m in range(iterStart, iterStart+NUMITERATIONS):
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
                for k in range(10, maxNumEpochs+1, 10): #CHANGE THESE TOGETHER
                    results[k] = ([],[])
                for m in range(iterStart, iterStart+NUMITERATIONS):
                    for i in range(10, maxNumEpochs+1, 10): #CHANGE THESE TOGETHER
                        images, labels = helpers.generateImageSample(40, rootDir, m, weightsUsed=maxNumEpochs)
                        train_images, train_labels, test_images, test_labels = [], [], [], []
                        for index in range(len(labels)):
                            if index % 40 < 20:
                                test_images.append(images[index])
                                test_labels.append(labels[index])
                            else:
                                train_images.append(images[index])
                                train_labels.append(labels[index])
                        invalidImageExistsFlag = True
                        while invalidImageExistsFlag:
                            try:
                                tf.keras.backend.clear_session()
                                resized_test = np.empty((len(test_images), 227, 227, 3))
                                for q in range(len(test_images)):
                                    resized_test[q] = tf.image.resize(tf.image.per_image_standardization(test_images[q]), (227,227))
                                network = DeepImageNetwork(numFeatures=1024)
                                resized_train = network.train(np.array(train_images), np.array(train_labels), numEpochs=i)
                                invalidImageExistsFlag = False
                            except:
                                print("invalid image found - resetting seed")
                                print(len(train_images), len(train_labels), len(test_images), len(test_labels))
                                images, labels = helpers.generateImageSample(40, rootDir, m, weightsUsed=maxNumEpochs)
                                train_images, train_labels, test_images, test_labels = [], [], [], []
                                for index in range(len(labels)):
                                    if index % 40 < 20:
                                        test_images.append(images[index])
                                        test_labels.append(labels[index])
                                    else:
                                        train_images.append(images[index])
                                        train_labels.append(labels[index])
                                continue
                        train_pred = network.predict(resized_train)
                        # print(train_pred.shape)
                        # print(train_pred[0])
                        test_pred = network.predict(resized_test)
                        # print(test_pred.shape)
                        # print(test_pred[0])
                        accuracyCount = 0
                        for j in range(len(train_labels)):
                            if train_labels[j] == np.argmax(train_pred[j]):
                                accuracyCount += 1
                        results[i][0].append(accuracyCount / (20*50.0))
                        print(accuracyCount / (20*50.0))
                        accuracyCount = 0
                        # print(test_labels)
                        for j in range(len(test_labels)):
                            if test_labels[j] == np.argmax(test_pred[j]):
                                accuracyCount += 1
                        results[i][1].append(accuracyCount / (20*50.0))
                        print(accuracyCount / (20*50.0))
                for k in results.keys():
                    print(str(k) + "," + str(sum(results[k][0]) / float(len(results[k][0]))) + "," + str(sum(results[k][1]) / float(len(results[k][1]))))
            elif int(userInput[1]) == 2:
                pass
                #TODO: implement??? (or use alternate weights???)
            if int(userInput[1]) == 0:
                record = open("../results/" + userInput[0] + "_" + userInput[1] + "_" + userInput[2] + "_" + str(m) + "_results.csv", "w")
                for iteration in results.keys():
                    record.write(str(iteration) + "," + ",".join(map(str, results[iteration])) + "\n")
                record.close()
            else:
                record = open("../results/" + userInput[0] + "_" + userInput[1] + "_" + userInput[2] + "_" + str(m) + "_results.csv", "w")
                for iteration in results.keys():
                    record.write(str(iteration) + "," + ",".join(map(str, results[iteration][0])) + "\n")
                    record.write(str(iteration) + "," + ",".join(map(str, results[iteration][1])) + "\n")
                record.close()

        # UserInput key:
        # 0 = test key [4 = new weight generation] #NOTE: this will only generate weights; they must be applied manually in another test!!!
        # 1 = modal key [0 = expert (weights), 1 = learned (features), 2 = mixed (weights)]
        # 2 = optional value to set limits of iterations (i.e., x to x+5) for parallelism
        elif int(userInput[0]) == 4:
            try:
                iterStart = int(userInput[2])
            except:
                iterStart = 0
            featureSelectionMode = int(userInput[1])
            if featureSelectionMode == 0:
                numFeatures = 85
            elif featureSelectionMode == 1:
                numFeatures = 1024
            else:
                numFeatures = 1109
            maxNumEpochs = 80
            for examplesPerAnimal in (10,20):
                results = {}
                for sigma in range(10, 91, 10):
                    for m in range(iterStart, iterStart+NUMITERATIONS):
                        if featureSelectionMode == 1 or featureSelectionMode == 2: #All learned, or mixed
                            images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir, m, 4, 0, 1)
                            invalidImageExistsFlag = True
                            while invalidImageExistsFlag:
                                try:
                                    tf.keras.backend.clear_session()
                                    network = DeepImageNetwork(numFeatures=features)
                                    resized_images = network.train(np.array(images), np.array(labels), numEpochs=50)
                                    invalidImageExistsFlag = False
                                except:
                                    print("invalid image found - resetting seed")
                                    images, labels = helpers.generateImageSample(examplesPerAnimal, rootDir, m, 4, 0, 1)
                                    continue
                            extractor = tf.keras.Model(inputs=network.model.input,\
                                                        outputs=network.model.layers[len(network.model.layers)-2].output)
                            outputs = extractor.predict(resized_images)

                        _, train, _ = Reader().readAwAForNN(rootDir)
                        inputs_control = np.empty((examplesPerAnimal*50, numFeatures))
                        labels = np.empty(examplesPerAnimal*50)
                        for a in range(50):
                            for e in range(examplesPerAnimal):
                                for f in range(numFeatures):
                                    if featureSelectionMode == 0 or (featureSelectionMode == 2 and f < 85):
                                        inputs_control[a*examplesPerAnimal+e][f] = train[a][f]
                                    elif featureSelectionMode == 1 or (featureSelectionMode == 2 and f >= 85):
                                        inputs_control[a*examplesPerAnimal+e][f] = outputs[a*examplesPerAnimal+e][f]
                                labels[a*examplesPerAnimal+e] = a
                        
                        if featureSelectionMode == 0:
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
                            if results.get(i) == None:
                                results[i] = {}
                            if results[i].get(sigma) == None:
                                results[i][sigma] = []
                            finalValue = (abs(accuracyCounts[0] - accuracyCounts[1])/float(len(labels)) + abs(accuracyCounts[0] - accuracyCounts[2])/float(len(labels)))/2.0
                            results[i][sigma].append(finalValue)
                        print(examplesPerAnimal, sigma, m)

                record = open("../results/" + userInput[0] + "_" + str(featureSelectionMode) + "_" + str(m) + "_results" + str(examplesPerAnimal) + ".csv", "w")
                for feat in results.keys():
                    for exampleNum in results[feat].keys():
                        record.write(str(feat) + "," + str(exampleNum) + "," + ",".join(map(str, results[feat][exampleNum])) + "\n")
                record.close()

        #===================================
        else:
            print("Error: unknown command")

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        raise(Exception("Error: incorrect number of arguments: " + str(len(sys.argv))))

    run(sys.argv[1])