class Feature:
    
    """
    Feature constructor
    - name = the name of the feature
    - value = the data associated with the feature (e.g., numeric or nominal data)
    - weight = the relative importance of the feature when calculating similarity
    - valueType = a keyword that helps define the similarity metric, which in turn controls how features are
                    compared between cases
    """
    def __init__(self, name, value, weight = 1, valueType = "inferred"):
        self.name = name
        self.value = value
        if valueType != "inferred":
            if valueType == "euclideanDistance":
                self.similarityMetric = lambda otherFeature: abs(self.value - otherFeature.value)
            else:
                self.similarityMetric = lambda otherFeature: 0.0 if self.value == otherFeature.value else 1.0
        elif type(value) is str:
            self.similarityMetric = lambda otherFeature: 0.0 if self.value == otherFeature.value else 1.0
        else:
            self.similarityMetric = lambda otherFeature: abs(self.value - otherFeature.value)
        self.weight = weight

    """
    To-string method for features
    """
    def __repr__(self):
        return "value = " + str(self.value) + ", weight = " + str(self.weight)

    """
    Hash method for features (a hash of the tuple of the name, value, and weight)
    """
    def __hash__(self):
        return hash((self.name, self.value, self.weight))

    """
    basic getter methods
    """
    def getWeight(self):
        return self.weight
    def getValue(self):
        return self.value

    """
    Setter methods
    """
    def setWeight(self, weight:float):
        self.weight = weight
    def setValue(self, value):
        self.value = value