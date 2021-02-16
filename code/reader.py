from case import Case
from feature import Feature
import numpy as np

class Reader:

    """
    Reads feature and classification information for an initial case base from a CSV file
    - filename = the path for the file to be read (generally used as a relative path from this file's directory)
    - classNames = an optional list of class names to associate with the incoming comma-separated values
    TODO: potentially consider being able to read class names from CSV columns
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
    Returns: predicate translation dictionary, feature training data, class translation dictionary (values used as classes for training)
    """
    def readAwAForNN(self):
        reader = open("data/awa2/predicates.txt", "r")
        lines = reader.readlines()
        predicates = {}
        for line in lines:
            words = line.strip().split("\t")
            predicates[words[1]] = int(words[0]) - 1
        reader.close()

        # reader = open("data/awa2/predicate-matrix-binary.txt", "r") # binary
        reader = open("data/awa2/predicate-matrix-continuous.txt", "r") # continuous
        lines = reader.readlines()
        train = []
        for line in lines:
            # words = line.strip().split(" ") # binary
            words = line.strip().split(" ") # continuous
            temp = []
            for value in words:
                if value != "":
                    temp.append(float(value))
            train.append(np.array(temp))
        reader.close()

        reader = open("data/awa2/classes.txt", "r")
        lines = reader.readlines()
        classes = {}
        for line in lines:
            words = line.strip().split("\t")
            classes[words[1]] = int(words[0]) - 1
        reader.close()

        return (predicates, train, classes)