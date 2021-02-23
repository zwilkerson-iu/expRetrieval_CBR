from case import Case

class CaseBase:

    """
    CaseBase constructor:
    - caseBaseSize = simple int representation of number of cases in case base (no len(cb.cases.keys()) necessary)
    - cases = dictionary of case objects indexed by case hash (case ID)
    - maxima = dictionary of maximum values in the case base, for purposes of normalizing Euclidean distance calculations
                (indexed by feature name)
    """
    def __init__(self, initialCases:dict = {}):

        self.caseBaseSize = 0
        self.cases = {}
        for caseID in initialCases.keys(): #Allowing shallow copies of the case objects, not the dictionary itself
            self.cases[caseID] = initialCases[caseID]
            self.caseBaseSize += 1
        self.maxima = {}
        #Other potential fields = _cbAltered, caseIdName, indexing, comparisonDict

    """
    Formal getter method for the case base (probably unnecessary)
    - caseID = case hash index
    Returns: the case object corresponding with caseID
    """
    def retrieveCase(self, caseID:int):
        return self.cases[caseID]
    
    """
    KNN implementation for the case base, using case similarity functions
    - case = query case
    - k = number of neighbors to consider
    Returns: a list of length k containing the nearest cases to the queried case, organized as tuples of
                caseID, similarityValue and sorted in ascending order of distance (i.e., increasing similarityValue)
    """
    def getKClosestCases(self, case:Case, k:int):
        bestCases = []
        for caseID in self.cases.keys():
            differenceVector = case.getDifferenceVector(self.retrieveCase(caseID))
            normalizedDistance = 0.0
            for differenceTuple in differenceVector.values():
                if differenceTuple[1] is None:
                    normalizedDistance += 1.0 #maximum possible dissimilarity
                else:
                    temp = differenceTuple[0].similarityMetric(differenceTuple[1]) * differenceTuple[1].weight
                    if differenceTuple[0].name in self.maxima:
                        temp /= self.maxima[differenceTuple[0].name]
                    normalizedDistance += temp
            
            if len(bestCases) < k or normalizedDistance < bestCases[-1][1]:
                bestCases.insert(0, (caseID, normalizedDistance))
                if len(bestCases) > k:
                    bestCases.pop()
                bestCases.sort(key = lambda x : x[1])
        return bestCases

    """
    1NN implementation that applies KNN method above
    - case = query case
    Returns: a tuple of the nearest case and the calculated distanace to that case
    """
    def getClosestCase(self, case:Case):
        cases = self.getKClosestCases(case, 1)
        if len(cases) == 0:
            return False
        else:
            return (self.cases[cases[0][0]], cases[0][1])

    """
    Case addition function for generating or modifying a case base
    - case = case to be added to the case base
    *Returns False if the case is already in the case base
    """
    def addCase(self, case:Case):
        caseID = hash(case)
        if caseID in self.cases:
            print("error adding case", caseID)
            return False
        else:
            for feature in case.features.keys():
                if (type(case.features[feature].value) == int or type(case.features[feature].value) == float) and \
                    (self.maxima.get(feature) == None or case.features[feature].value > self.maxima[feature]):
                    self.maxima[feature] = case.features[feature].value
            self.cases[caseID] = case
            self.caseBaseSize += 1
    
    # def alterCase(self, caseID, newFeatureValues)

    """
    Case removal feature (e.g., for case base maintenance)
    - caseID = case hash for case to be removed
    Returns: the case object being removed from the case base, or False if it was not in the case base
    """
    def removeCase(self, caseID:int):
        try:
            self.caseBaseSize -= 1
            case = self.cases.pop(caseID)
            for feature in case.features.keys():
                if self.maxima[feature] == case.features[feature].value:
                    self.maxima[feature] = self.findMaximaFromCaseBase(feature)
            return case
        except KeyError:
            print("Error removing case")
            return False

    """
    Maxima generation by feature that queries the case base (used for removing a case containing a maximum value)
    - featureName = name of the feature to query for maximum values
    Returns: the maximum value found for the feature in the case base
    """
    def findMaximaFromCaseBase(self, featureName:str):
        maxVal = 0
        for caseID in self.cases.keys():
            temp = self.cases[caseID].features.get(featureName)
            if temp != None and temp.value > maxVal:
                maxVal = temp.value
        return maxVal