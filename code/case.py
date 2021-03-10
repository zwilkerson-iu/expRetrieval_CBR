from feature import Feature

class Case:

    """
    Case constructor
    - featureAttrbutes = dictionary of existing features (esp. if copying from another case, otherwise usually empty)
    - result = class of the case (single value if performing regression, class-confidence tuple if performing classification)
    """
    def __init__(self, featureAttributes:dict, result):
        self.features = {}
        for key, feature in featureAttributes.items():
            self.features[key] = feature
        self.result = result

    """
    Hash function for case (just a sum of the hash values for its attributes)
    """
    # def __hash__(self):
    #     hashSum = 0.0
    #     for feature in self.features.values():
    #         hashSum += hash(feature)
    #     return hash(hashSum)

    """
    To-string representation for a case, highlighting features and the result
    """
    def __repr__(self):
        featureString = "Features:\n"
        for feature in self.features.keys():
            temp = str(self.features[feature])
            featureString += feature + " : " + temp + "\n"
        if type(self.result) is not tuple:
            featureString += "Result:\n" + str(self.result)+ "\n"
        else:
            featureString += "Result:\n" + str(self.result[0]) + " (" + str(100*self.result[1]) + "%)\n"
        return super().__repr__() + "\n" + featureString

    """
    Similarity foundational function that returns a dictionary containing tuples of all pairs of differing
        features between the self case and the other case to which it is compared
    - otherCase = the case to which the calling case is compared
    Returns: a dictionary containing tuples of all features that are different between the two cases,
                indexed by feature name
    """
    def getDifferenceVector(self, otherCase):
        differenceVector = {}
        for feature in self.features.values():
            try:
                temp = feature.similarityMetric(otherCase.features[feature.name])
                if temp != 0:
                    differenceVector[feature.name] = (feature, otherCase.features[feature.name])
            except KeyError:
                differenceVector[feature.name] = (feature, None)
        return differenceVector

    """
    Feature addition to make case implementation more dynamic
    - feature = feature object to be added
    *Returns False if the feature already existed for this case
    """
    def addNewFeature(self, feature:Feature):
        try:
            _ = self.features[feature.name]
            return False
        except KeyError:
            self.features[feature.name] = feature

    """
    Basic feature getter functions (potentially unnecessary)
    - featureName = the name of the feature to be accessed
    Returns: the feature corresponding with featureName, or False if it does not exist
    """
    def getFeature(self, featureName:str):
        try:
            return self.features[featureName]
        except KeyError:
            return False
    def getFeatures(self):
        returnValue = {}
        for key, feature in self.features.items():
            returnValue[key] = feature
        return returnValue

    """
    Base setter function
    """
    def editFeature(self, featureName:str, newValue):
        try:
            self.features[featureName].setValue(newValue)
        except KeyError:
            return False

    """
    Basic feature removal
    - featureName = the name of the feature to be removed
    Returns the feature to be removed for this case, or False if did not exist
    """
    def removeFeature(self, featureName:str):
        try:
            return self.features.pop(featureName)
        except KeyError:
            return False