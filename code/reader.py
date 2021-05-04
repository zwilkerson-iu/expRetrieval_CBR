from case import Case
from feature import Feature
import numpy as np
import os
import statistics
import matplotlib.pyplot as plt

class Reader:

    """
    Reads feature and classification information for an initial case base from a CSV file
    - filename = the path for the file to be read (generally used as a relative path from this file's directory)
    - classNames = an optional list of class names to associate with the incoming comma-separated values
    Returns: a list of case objects that can be used to create/add to a case base object
    """
    def readCSVCases(self, filename:str, classNames:list = None, regressionFlag:bool = False):
        returnValue = []
        readFile = open(filename, "r")
        lines = readFile.readlines()
        for line in lines:
            words = line.strip().split(",")
            if not regressionFlag:
                newCase = Case({}, (words[-1], 1.0)) #assumes last column is classification
            else:
                newCase = Case({}, words[-1]) # assumes last column is regression value
            if classNames != None and (len(words) == len(classNames) + 1):
                for i in range(len(words) - 1):
                    try:
                        newCase.addNewFeature(Feature(classNames[i], float(words[i]), 1, "euclideanDistance"))
                    except ValueError:
                        newCase.addNewFeature(Feature(classNames[i], words[i], 1, "match"))
            else:
                for i in range(len(words) - 1):
                    try:
                        newCase.addNewFeature(Feature("class" + str(i), float(words[i]), 1, "euclideanDistance"))
                    except ValueError:
                        newCase.addNewFeature(Feature("class" + str(i), words[i], 1, "match"))
            returnValue.append(newCase)
        return returnValue

    """
    Reads feature data for the AwA2 dataset a list of cases
    - filename = the feature data file path to read
    - classFilename = the path for class labels to populate result fields for the new case objects
    - predicateFilename = optional feature names for new cases
    Returns: a list of case objects pertaining to the read feature data
    """
    def readAwADataFromTxt(self, filename:str, classFilename:str, predicateFilename:str = None):
        reader = open(classFilename, "r")
        lines = reader.readlines()
        indices = []
        for line in lines:
            words = line.strip().split("\t")
            indices.append(words[1])
        reader.close()

        if predicateFilename is not None:
            reader = open(predicateFilename, "r")
            lines = reader.readlines()
            predicates = []
            for line in lines:
                words = line.strip().split("\t")
                predicates.append(words[1])
            reader.close()
        
        retVal = []
        reader = open(filename, "r")
        lines = reader.readlines()
        for i in range(len(lines)):
            words = lines[i].strip().split(" ")
            newCase = Case({}, (indices[i], 1.0)) #TODO: account for regression
            for r in range(len(words)-1, -1, -1): # remove blank values if using continuous
                if words[r] == "":
                    words.pop(r)
            for j in range(len(words)):
                if predicateFilename is not None:
                    newCase.addNewFeature(Feature(predicates[j], float(words[j]), 1, "match"))
                else:
                    newCase.addNewFeature(Feature("feature" + str(j), float(words[j]), 1, "match"))
            retVal.append(newCase)
        reader.close()

        return retVal
    """
    Reads feature data from the AwA dataset for neural network training to generate feature weights
    - rootDir = root directory of the system (testing parameter)
    Returns: predicate translation dictionary, feature training data, class translation dictionary (values used as classes for training)
    """
    def readAwAForNN(self, rootDir:str):
        reader = open(rootDir + "awa2/predicates.txt", "r")
        lines = reader.readlines()
        predicates = {}
        for line in lines:
            words = line.strip().split("\t")
            predicates[words[1]] = int(words[0]) - 1
        reader.close()

        reader = open(rootDir + "awa2/predicate-matrix-continuous.txt", "r")
        lines = reader.readlines()
        train = []
        for line in lines:
            words = line.strip().split(" ") # continuous
            temp = []
            for value in words:
                if value != "":
                    temp.append(float(value))
            train.append(np.array(temp))
        reader.close()

        reader = open(rootDir + "awa2/classes.txt", "r")
        lines = reader.readlines()
        classes = {}
        for line in lines:
            words = line.strip().split("\t")
            classes[words[1]] = int(words[0]) - 1
        reader.close()

        return (predicates, train, classes)

    """
    Reads feature raw data from results files and compiles into condensed/organized final results files
    - rootDir = root directory of the system (testing parameter)
    - arg1 = test definition argument, on [0,4]
    - arg2 = test specification argument
    - arg3 = optional argument when running later iterations of the same test (either to reduce stdev or test extra ideas without revamping file naming)
    """
    def analyzeData(self, rootDir:str, arg1:int, arg2:int, arg3:int = 0):
        if arg1 == 0 or arg1 == 2 or arg1 == 4:
            results = {10:{}}
        elif arg1 == 1:
            results = {10:[]}
        else:
            if arg2 == 1:
                results = {"train":{}, "test":{}}
            else:
                results = {"train":{}}
        files = []
        if arg1 == 0 or arg1 == 1 or arg1 == 2:
            for filename in os.listdir(rootDir):
                if filename[0] == str(arg1) and filename[4] == str(arg2):
                    if filename[6] == 'r':
                        files.append(filename)
                    else:
                        s = 6
                        e = 7
                        while filename[e] != "_":
                            e += 1
                        if int(filename[s:e]) >= arg3 and int(filename[s:e]) < arg3+30:
                            files.append(filename)
        else:
            for filename in os.listdir(rootDir):
                if filename[0] == str(arg1) and filename[2] == str(arg2):
                    if arg1 == 3:
                        if filename[6] == 'r':
                            files.append(filename)
                        else:
                            s = 6
                            e = 7
                            while filename[e] != "_":
                                e += 1
                            if int(filename[s:e]) >= arg3 and int(filename[s:e]) < arg3+30:
                                files.append(filename)
                    else:
                        if filename[4] == 'r':
                            files.append(filename)
                        else:
                            s = 4
                            e = 5
                            while filename[e] != "_":
                                e += 1
                            if int(filename[s:e]) >= arg3 and int(filename[s:e]) < arg3+30:
                                files.append(filename)
        for filename in files:
            if arg1 == 0 or arg1 == 1 or arg1 == 2:
                exampleCount = int(filename[-6:-4])
                reader = open(rootDir + filename, "r")
                if arg1 == 0:
                    if results[exampleCount].get(int(filename[2])) is None:
                        results[exampleCount][int(filename[2])] = []
                    for line in reader.readlines():
                        words = line.split(",")
                        if words[0] != "average" and words[0] != "stdev":
                            results[exampleCount][int(filename[2])].append(float(words[1]))
                elif arg1 == 1:
                    for line in reader.readlines():
                        words = line.split(",")
                        if words[0] != "average" and words[0] != "stdev":
                            results[exampleCount].append(float(words[1]))
                else:
                    if results[exampleCount].get(int(filename[2])) is None:
                        results[exampleCount][int(filename[2])] = {}
                    for line in reader.readlines():
                        words = line.split(",")
                        if words[0] != "average" and words[0] != "stdev":
                            for i, j in ((10, 1),(30, 2),(50, 3),(70, 4),(90, 5),(100, 6)):
                                if results[exampleCount][int(filename[2])].get(i) is None:
                                    results[exampleCount][int(filename[2])][i] = []
                                results[exampleCount][int(filename[2])][i].append(float(words[j]))
            elif arg1 == 4:
                exampleCount = int(filename[-6:-4])
                reader = open(rootDir + filename, "r")
                for line in reader.readlines():
                    words = line.split(",")
                    if words[0] != "average" and words[0] != "stdev":
                        if results[exampleCount].get(int(words[0])) is None:
                            results[exampleCount][int(words[0])] = {}
                        if results[exampleCount][int(words[0])].get(int(words[1])) is None:
                            results[exampleCount][int(words[0])][int(words[1])] = []
                        for i in range(2, 7):
                            results[exampleCount][int(words[0])][int(words[1])].append(float(words[i]))
            elif arg1 == 3:
                reader = open(rootDir + filename, "r")
                lines = reader.readlines()
                for l in range(len(lines)):
                    words = lines[l].split(",")
                    if words[0] != "average" and words[0] != "stdev":
                        if l % 2 == 0 or arg2 == 0 or arg2 == 2:
                            if results["train"].get(int(words[0])) is None:
                                results["train"][int(words[0])] = []
                            for i in range(1, 6):
                                results["train"][int(words[0])].append(float(words[i]))
                        else:
                            if results["test"].get(int(words[0])) is None:
                                results["test"][int(words[0])] = []
                            for i in range(1, 6):
                                results["test"][int(words[0])].append(float(words[i]))
        if arg1 == 0:
            for example in results.keys():
                record = open(rootDir + "finalResults/" + str(arg1) + "_" + str(arg2) + "_" + str(arg3) + "_" + str(example) + "_finalResults.csv", "w")
                record.write("rand. mult.,average,stdev,raw values\n")
                for rand in results[example].keys():
                    stdev = statistics.stdev(results[example][rand])
                    ave = sum(results[example][rand]) / float(len(results[example][rand]))
                    record.write(str(rand) + "," + str(ave) + "," + str(stdev) + "," + ",".join(map(str, results[example][rand])) + "\n")
                record.close()
        elif arg1 == 1:
            for example in results.keys():
                try:
                    record = open(rootDir + "finalResults/" + str(arg1) + "_" + str(arg2) + "_" + str(arg3) + "_" + str(example) + "_finalResults.csv", "w")
                    record.write("average,stdev,raw values\n")
                    stdev = statistics.stdev(results[example])
                    ave = sum(results[example]) / float(len(results[example]))
                    record.write(str(ave) + "," + str(stdev) + "," + ",".join(map(str, results[example])) + "\n")
                    record.close()
                except:
                    continue
        elif arg1 == 2:
            for example in results.keys():
                record = open(rootDir + "finalResults/" + str(arg1) + "_" + str(arg2) + "_" + str(arg3) + "_" + str(example) + "_finalResults.csv", "w")
                record.write("rand. mult.,expert %,average,stdev,raw values\n")
                for rand in results[example].keys():
                    for frac in results[example][rand].keys():
                        stdev = statistics.stdev(results[example][rand][frac])
                        ave = sum(results[example][rand][frac]) / float(len(results[example][rand][frac]))
                        record.write(str(rand) + "," + str(frac) + "," + str(ave) + "," + str(stdev) + "," + ",".join(map(str, results[example][rand][frac])) + "\n")
                record.close()
        elif arg1 == 3:
            record = open(rootDir + "finalResults/" + str(arg1) + "_" + str(arg2) + "_" + str(arg3) + "_finalResults.csv", "w")
            for word in results.keys():
                record.write("epochs,average,stdev,raw values\n")
                for epoch in results[word].keys():
                    stdev = statistics.stdev(results[word][epoch])
                    ave = sum(results[word][epoch]) / float(len(results[word][epoch]))
                    record.write(str(epoch) + "," + str(ave) + "," + str(stdev) + "," + ",".join(map(str, results[word][epoch])) + "\n")
                record.write("\n")
            record.close()
        elif arg1 == 4:
            for example in results.keys():
                record = open(rootDir + "finalResults/" + str(arg1) + "_" + str(arg2) + "_" + str(arg3) + "_" + str(example) + "_finalResults.csv", "w")
                for multiplier in results[example][0].keys():
                    record.write("feature,multiplier,average,stdev,raw values\n")
                    for feature in results[example].keys():
                        stdev = statistics.stdev(results[example][feature][multiplier])
                        ave = sum(results[example][feature][multiplier]) / float(len(results[example][feature][multiplier]))
                        record.write(str(feature) + "," + str(multiplier) + "," + str(ave) + "," + str(stdev) + "," + ",".join(map(str, results[example][feature][multiplier])) + "\n")
                    record.write("\n")
                record.close()

    #TODO: documentation
    def createFigure(self, unweighted:bool, rootDir:str, centroid:int, centroid2 = 8):
        dataPoints = {"KE":{}, "CFE":{}, "Both":{}}
        t = {"0":"KE", "1":"CFE", "2":"Both"}
        titles = ("A", "B", "C", "D", "E", "F")
        if unweighted:
            files1 = ("0_0_0_10_finalResults.csv", "1_0_0_10_finalResults.csv", "2_0_0_10_finalResults.csv")
            files2 = ("0_0_100_10_finalResults.csv", "1_0_100_10_finalResults.csv", "2_0_100_10_finalResults.csv")
            #Initial data
            for filename in files1:
                reader = open(rootDir + "finalResults/" + filename, "r")
                lines = reader.readlines()
                if filename[0] == "0":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if abs(int(words[0]) - centroid) <= 1:
                            if dataPoints[t[filename[0]]].get(int(words[0])) is None:
                                dataPoints[t[filename[0]]][int(words[0])] = []
                            for i in range(3, len(words)):
                                dataPoints[t[filename[0]]][int(words[0])].append(float(words[i]))
                elif filename[0] == "1":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        for key in dataPoints["KE"].keys():
                            dataPoints[t[filename[0]]][key] = []
                            for i in range(2, len(words)):
                                    dataPoints[t[filename[0]]][key].append(float(words[i]))
                elif filename[0] == "2":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if abs(int(words[0]) - centroid) <= 1 and int(words[1]) == 100:
                            if dataPoints[t[filename[0]]].get(int(words[0])) is None:
                                dataPoints[t[filename[0]]][int(words[0])] = []
                            for i in range(4, len(words)):
                                dataPoints[t[filename[0]]][int(words[0])].append(float(words[i]))
            #plots
            figure = plt.figure(constrained_layout=True)
            plots = figure.subplots(2, 3, squeeze=False)
            for j in range(3):
                stdevs = (statistics.stdev(dataPoints["KE"][tuple(dataPoints["KE"].keys())[j]]),
                            statistics.stdev(dataPoints["CFE"][tuple(dataPoints["CFE"].keys())[j]]),
                            statistics.stdev(dataPoints["Both"][tuple(dataPoints["Both"].keys())[j]]))
                means = (statistics.mean(dataPoints["KE"][tuple(dataPoints["KE"].keys())[j]]),
                            statistics.mean(dataPoints["CFE"][tuple(dataPoints["CFE"].keys())[j]]),
                            statistics.mean(dataPoints["Both"][tuple(dataPoints["Both"].keys())[j]]))
                # for a, b in enumerate(means):
                #     plots[0, j].text(b, a, str(round(b, 3)), color='blue', fontweight='bold')
                plots[0, j].bar(["KE", "CFE", "Both"], means, yerr=stdevs, capsize=4.0)
                plots[0, j].set_title(titles[j])
                plots[0, j].set_ylim([0.0,0.75])
            #More trials
            for filename in files2:
                reader = open(rootDir + "finalResults/" + filename, "r")
                lines = reader.readlines()
                if filename[0] == "0":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if abs(int(words[0]) - centroid) <= 1:
                            for i in range(3, len(words)):
                                dataPoints[t[filename[0]]][int(words[0])].append(float(words[i]))
                elif filename[0] == "1":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        for key in dataPoints["KE"].keys():
                            for i in range(2, len(words)):
                                    dataPoints[t[filename[0]]][key].append(float(words[i]))
                elif filename[0] == "2":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if abs(int(words[0]) - centroid) <= 1 and int(words[1]) == 100:
                            for i in range(4, len(words)):
                                dataPoints[t[filename[0]]][int(words[0])].append(float(words[i]))
            #More plots
            for j in range(3):
                stdevs = (statistics.stdev(dataPoints["KE"][tuple(dataPoints["KE"].keys())[j]]),
                            statistics.stdev(dataPoints["CFE"][tuple(dataPoints["CFE"].keys())[j]]),
                            statistics.stdev(dataPoints["Both"][tuple(dataPoints["Both"].keys())[j]]))
                means = (statistics.mean(dataPoints["KE"][tuple(dataPoints["KE"].keys())[j]]),
                            statistics.mean(dataPoints["CFE"][tuple(dataPoints["CFE"].keys())[j]]),
                            statistics.mean(dataPoints["Both"][tuple(dataPoints["Both"].keys())[j]]))
                # for a, b in enumerate(means):
                #     plots[1, j].text(b, a, str(round(b, 3)), color='blue', fontweight='bold')
                plots[1, j].bar(["KE", "CFE", "Both"], means, yerr=stdevs, capsize=4.0)
                plots[1, j].set_title(titles[3+j])
                plots[1, j].set_ylim([0.0,0.75])
            figure.savefig(rootDir + "finalResults/unweightedFigure.png")

        else:
            dataPointsRELU = {"KE":{}, "CFE":{}, "Both":{}}
            files = ("0_1_0_10_finalResults.csv", "0_1_30_10_finalResults.csv", "1_1_0_10_finalResults.csv",
                        "1_1_30_10_finalResults.csv", "2_1_0_10_finalResults.csv", "2_1_30_10_finalResults.csv")
            for filename in files:
                reader = open(rootDir + "finalResults/" + filename, "r")
                lines = reader.readlines()
                if filename[0] == "0":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if filename[4] == "0":
                            if abs(int(words[0]) - centroid) <= 1:
                                if dataPoints[t[filename[0]]].get(int(words[0])) is None:
                                    dataPoints[t[filename[0]]][int(words[0])] = []
                                for i in range(3, len(words)):
                                    dataPoints[t[filename[0]]][int(words[0])].append(float(words[i]))
                        else:
                            if abs(int(words[0]) - centroid2) <= 1:
                                if dataPointsRELU[t[filename[0]]].get(int(words[0])) is None:
                                    dataPointsRELU[t[filename[0]]][int(words[0])] = []
                                for i in range(3, len(words)):
                                    dataPointsRELU[t[filename[0]]][int(words[0])].append(float(words[i]))
                elif filename[0] == "1":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if filename[4] == "0":
                            for key in dataPoints["KE"].keys():
                                dataPoints[t[filename[0]]][key] = []
                                for i in range(2, len(words)):
                                        dataPoints[t[filename[0]]][key].append(float(words[i]))
                        else:
                            for key in dataPointsRELU["KE"].keys():
                                dataPointsRELU[t[filename[0]]][key] = []
                                for i in range(2, len(words)):
                                        dataPointsRELU[t[filename[0]]][key].append(float(words[i]))
                elif filename[0] == "2":
                    for l in range(1, len(lines)):
                        line = lines[l]
                        words = line.strip().split(",")
                        if filename[4] == "0":
                            if abs(int(words[0]) - centroid) <= 1 and int(words[1]) == 100:
                                if dataPoints[t[filename[0]]].get(int(words[0])) is None:
                                    dataPoints[t[filename[0]]][int(words[0])] = []
                                for i in range(4, len(words)):
                                    dataPoints[t[filename[0]]][int(words[0])].append(float(words[i]))
                        else:
                            if abs(int(words[0]) - centroid2) <= 1 and int(words[1]) == 100:
                                if dataPointsRELU[t[filename[0]]].get(int(words[0])) is None:
                                    dataPointsRELU[t[filename[0]]][int(words[0])] = []
                                for i in range(4, len(words)):
                                    dataPointsRELU[t[filename[0]]][int(words[0])].append(float(words[i]))
            figure = plt.figure(constrained_layout=True)
            plots = figure.subplots(2, 3, squeeze=False)
            for j in range(3):
                stdevs = (statistics.stdev(dataPoints["KE"][tuple(dataPoints["KE"].keys())[j]]),
                            statistics.stdev(dataPoints["CFE"][tuple(dataPoints["CFE"].keys())[j]]),
                            statistics.stdev(dataPoints["Both"][tuple(dataPoints["Both"].keys())[j]]))
                means = (statistics.mean(dataPoints["KE"][tuple(dataPoints["KE"].keys())[j]]),
                            statistics.mean(dataPoints["CFE"][tuple(dataPoints["CFE"].keys())[j]]),
                            statistics.mean(dataPoints["Both"][tuple(dataPoints["Both"].keys())[j]]))
                # for a, b in enumerate(means):
                #     plots[1, j].text(b, a, str(round(b, 3)), color='blue', fontweight='bold')
                plots[0, j].bar(["KE", "CFE", "Both"], means, yerr=stdevs, capsize=4.0)
                plots[0, j].set_title(titles[j])
                plots[0, j].set_ylim([0.0,0.75])
            for j in range(3):
                stdevs = (statistics.stdev(dataPointsRELU["KE"][tuple(dataPointsRELU["KE"].keys())[j]]),
                            statistics.stdev(dataPointsRELU["CFE"][tuple(dataPointsRELU["CFE"].keys())[j]]),
                            statistics.stdev(dataPointsRELU["Both"][tuple(dataPointsRELU["Both"].keys())[j]]))
                means = (statistics.mean(dataPointsRELU["KE"][tuple(dataPointsRELU["KE"].keys())[j]]),
                            statistics.mean(dataPointsRELU["CFE"][tuple(dataPointsRELU["CFE"].keys())[j]]),
                            statistics.mean(dataPointsRELU["Both"][tuple(dataPointsRELU["Both"].keys())[j]]))
                # for a, b in enumerate(means):
                #     plots[1, j].text(b, a, str(round(b, 3)), color='blue', fontweight='bold')
                plots[1, j].bar(["KE", "CFE", "Both"], means, yerr=stdevs, capsize=4.0)
                plots[1, j].set_title(titles[3+j])
                plots[1, j].set_ylim([0.0,0.75])
            figure.savefig(rootDir + "finalResults/weightedFigure.png")