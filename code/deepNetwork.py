import tensorflow as tf
import numpy as np

class DeepImageNetwork:
    
    """
    Network constructor
    - model = optional custom model for more complex tasks later
    - photoDim = tuple containing x, y dimensions of the input photos for the network
    - numOutputs = number of outputs for the network (for AwA2 dataset, this should be 50)
    - activation = the activation function for hidden layers
    """
    def __init__(self, model:tf.keras.Sequential = None, photoDim:tuple = (1, 1), numOutputs:int = 1, numFeatures = 128, activation = "relu"):
        if model is not None:
            self.model = model
        else:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(photoDim[0], photoDim[1], 3)),
                tf.keras.layers.Dense(numFeatures, activation=activation, use_bias=False),
                tf.keras.layers.Dense(numFeatures, activation=activation, use_bias=False),
                tf.keras.layers.Dense(numFeatures, activation=activation, use_bias=False),
                tf.keras.layers.Dense(numFeatures, activation=activation, use_bias=False),
                tf.keras.layers.Dense(numOutputs, use_bias=False)
            ])

            self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    """
    Network training algorithm
    - train_images = the set of training images for the network
    - train_labels = the set of labels for weight updates during the training process
    - numEpochs = the number of training iterations or epochs to perform
    Returns: resized images (in case needed for later testing)
    """
    def train(self, train_images:np.array, train_labels:np.array, numEpochs:int = 10):
        resized_images = []
        for i in range(len(train_images)):
            resized_images.append(tf.keras.preprocessing.image.smart_resize(train_images[i], (1200,1200)))
        resized_images = np.array(resized_images)
        self.model.fit(resized_images, train_labels, epochs=numEpochs, verbose=0)
        # self.model.fit(resized_images, train_labels, epochs=numEpochs, verbose=1)
        return resized_images

    """
    Prediction algorithm for trained neural network
    - test_images = list containing 1 or more images to test
    Returns: a list of lists containing prediction values for each output, per test image
    """
    def predict(self, test_images:np.array):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        return probability_model.predict(test_images)

class FeatureNetwork:

    """
    Network constructor
    - model = optional custom model for more complex tasks later
    - numInputs = number of inputs for the network (i.e., number of features)
    - numOutputs = number of outputs for the network (for AwA2 dataset, this should be 50)
    """
    def __init__(self, model:tf.keras.Sequential = None, numInputs:int = 1, numOutputs:int = 1):
        if model is not None:
            self.model = model
        else:
            self.model = tf.keras.Sequential([
                # tf.keras.layers.Dense(numInputs, activation="relu"), #Is this necessary, or extra layer?
                # tf.keras.layers.Dense(numInputs, activation="sigmoid"),
                tf.keras.layers.Dense(numOutputs, use_bias=False) #TODO: test use_bias = False
            ])

            self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    """
    Network training algorithm
    - train_featuresList = the set of training features for the network
    - train_labels = the set of labels for weight updates during the training process
    - numEpochs = the number of training iterations or epochs to perform
    """
    def train(self, train_featuresList:np.array, train_labels:np.array, numEpochs:int = 10):
        self.model.fit(train_featuresList, train_labels, epochs=numEpochs, verbose=0)
        # print(self.model.trainable_weights)
        # print(self.model.trainable_weights[0].numpy())

    """
    Prediction algorithm for trained neural network
    - test_list = list containing 1 or more examples to test
    Returns: a list of lists containing prediction values for each output, per test image
    """
    def predict(self, test_list:np.array):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        return probability_model.predict(test_list)