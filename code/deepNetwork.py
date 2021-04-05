import tensorflow as tf
import numpy as np

class DeepImageNetwork:
    
    """
    Network constructor
    - model = optional custom model for more complex tasks later
    - photoDim = tuple containing x, y dimensions of the input photos for the network
    - numOutputs = number of outputs for the network (for AwA2 dataset, this should be 50)
    - numFeatures = number of features to be trained (i.e., the size of the Dense layers)
    - activation = the activation function for hidden layers
    """
    def __init__(self, model:tf.keras.Sequential = None, photoDim:tuple = (227, 227), numOutputs:int = 50, numFeatures:int = 4096, activation = "relu"):
        if model is not None:
            self.model = model
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
                tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(numFeatures, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(numFeatures, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(numOutputs, activation='softmax', use_bias=False)
            ])
            self.model.compile(loss='sparse_categorical_crossentropy', 
                                optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

    """
    Network training algorithm
    - train_images = the set of training images for the network
    - train_labels = the set of labels for weight updates during the training process
    - numFeatures = number of features to be trained (i.e., the size of the Dense layers in the model)
    - numEpochs = the number of training iterations or epochs to perform
    Returns: resized images (in case needed for later testing)
    """
    def train(self, train_images:np.array, train_labels:np.array, numFeatures:tuple = (227, 227), numEpochs:int = 20):
        resized_images = np.empty((len(train_images), numFeatures[0], numFeatures[1], 3))
        for i in range(len(train_images)):
            resized_images[i] = tf.image.resize(tf.image.per_image_standardization(train_images[i]), (227,227))
        self.model.fit(resized_images, train_labels, epochs=numEpochs, verbose=1)
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
                tf.keras.layers.Dense(numOutputs, use_bias=False, activation="relu") #TODO: test use_bias = False
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

    """
    Prediction algorithm for trained neural network
    - test_list = list containing 1 or more examples to test
    Returns: a list of lists containing prediction values for each output, per test image
    """
    def predict(self, test_list:np.array):
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        return probability_model.predict(test_list)