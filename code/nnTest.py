import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import random
from skimage.io import imread

rootDir = "../../expRetrieval_CBR_data/"

def test(nodesPerHiddenLayer:int, numImagesPerAnimal:int):
    images = []
    labels = []
    classes = os.listdir(rootDir + "awa2/JPEGImages")
    for index in range(len(classes)):
        animal = classes[index]
        imageFiles = os.listdir(rootDir + "awa2/JPEGImages/" + animal)
        imageTemps = random.sample(imageFiles, numImagesPerAnimal)
        # print(animal + "," + ",".join(x for x in imageTemps))
        for filepath in imageTemps:
            temp = imread(rootDir + "awa2/JPEGImages/" + animal + "/" + filepath, as_gray = False)
            images.append(np.asarray(temp))
        labels = labels + [index] * numImagesPerAnimal

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(200, 200, 3)),
        tf.keras.layers.Dense(nodesPerHiddenLayer, activation="sigmoid", use_bias=False),
        #tf.keras.layers.Dense(nodesPerHiddenLayer, activation="relu", use_bias=False),
        #tf.keras.layers.Dense(nodesPerHiddenLayer, activation="relu", use_bias=False),
        #tf.keras.layers.Dense(nodesPerHiddenLayer, activation="relu", use_bias=False),
        tf.keras.layers.Dense(50, use_bias=False)
    ])
    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'])

    resized_images = []
    # images = np.array(images)
    labels = np.asarray(labels)
    # print(labels)
    for i in range(len(images)):
        if i % 10 == 0:
            print(i)
        try:
            resized_images.append(tf.keras.preprocessing.image.smart_resize(images[i], (200,200)))
        except:
            continue
    resized_images = np.asarray(resized_images)
    # esized_images = tf.image.rgb_to_grayscale(resized_images)
    print(resized_images.shape, labels.shape)
    print(resized_images[0])
    model.fit(resized_images, labels, epochs=10, verbose=1)
    # return resized_images

    extractor = tf.keras.Model(inputs=model.input,\
                outputs=model.layers[len(model.layers)-2].output)
    outputs = extractor.predict(resized_images)
    print(outputs[:20])

def alex(numImagesPerAnimal:int):
    flag = True
    while flag:
        images = [] #TODO: make this a numpy array to begin with
        labels = np.empty(50 * numImagesPerAnimal)
        classes = os.listdir(rootDir + "awa2/JPEGImages")
        for index in range(len(classes)):
            animal = classes[index]
            imageFiles = os.listdir(rootDir + "awa2/JPEGImages/" + animal)
            imageTemps = random.sample(imageFiles, numImagesPerAnimal)
            # print(animal + "," + ",".join(x for x in imageTemps))
            for f in range(len(imageTemps)):
                temp = imread(rootDir + "awa2/JPEGImages/" + animal + "/" + imageTemps[f], as_gray = False)
                images.append(temp)
                labels[index*numImagesPerAnimal+f] = index

        resized_images = np.empty((50 * numImagesPerAnimal, 227, 227, 3))
        try:
            for i in range(len(images)):
                resized_images[i] = tf.image.resize(tf.image.per_image_standardization(images[i]), (227,227))
                print(i)
            flag = False
        except:
            continue

    #train_ds = tf.data.Dataset.from_tensor_slices((np.asarray(images), np.asarray(labels)))
    #train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()

    def process_images(image):
    # def process_images(image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images from 32x32 to 277x277
        image = tf.image.resize(image, (227,227))
        return image
        # return image, label

    # train_ds = (train_ds
    #               .map(process_images)
    #               .shuffle(buffer_size=train_ds_size)
    #               .batch(batch_size=32, drop_remainder=True))

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(50, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])

    # model.fit(train_ds, epochs=10, verbose=1)
    model.fit(resized_images, labels, epochs=70, verbose=1)

alex(20)