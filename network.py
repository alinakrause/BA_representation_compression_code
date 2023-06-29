
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import LambdaCallback
from binning import binning
from mutual_information import mutualInformation
from information_flow import information_flow
from plots import info_plane
import mysql.connector
from plots import information_plane2
from plots import plot_epochs, plot_for_epoch
from dataset import loadData, prepData

layer_outputs = []





# with a Sequential model
def get_output(model,x_train):
    outputs = []
    functors = []
    layer_activations = []

    # all output tensors from each layer in the model
    for layer in model.layers:
        outputs.append(layer.output)

    # evaluate the output tensor given an input
    for out in outputs:
        functor = K.function([model.input], [out])
        functors.append(functor)

    # extract layer activations for input data
    for f in functors:
        activation = f([x_train])
        layer_activations.append(activation)

    layer_outputs.append(layer_activations)


if __name__ == "__main__":
    x,y = loadData()
    x_train, x_test, y_train, y_test = prepData(x,y)


    shape=len(x_train[0])
    layers = [12,10,7,5,4,3,2,1]
    activationFuctions = ['tanh','relu']
    activation= activationFuctions[1]
    batch_size = 180
    epochs = 150  # 100 #500





    model = keras.models.Sequential([
        keras.layers.Dense(layers[0], activation=activation,input_shape=(shape,)),
       # keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(layers[1], activation=activation),
        keras.layers.Dense(layers[2], activation=activation),
        keras.layers.Dense(layers[3], activation=activation),
        keras.layers.Dense(layers[4], activation=activation),
        keras.layers.Dense(layers[5], activation=activation),
        keras.layers.Dense(layers[6], activation=activation),
        keras.layers.Dense(layers[7], activation='sigmoid'),
    ])

    print(model.summary())

    loss = tf.keras.losses.BinaryCrossentropy() # from_logits=True
    #loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


    #optim = tf.keras.optimizers.experimental.SGD(learning_rate=0.006) #0.01
    optim = tf.keras.optimizers.Adam(learning_rate=0.0001)
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    # at the end of every epoch
    callback = LambdaCallback(on_epoch_end = lambda batch, logs:get_output(model, x_train))

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, callbacks =[callback])

    # evaulate
    model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)
    #print(x)
    #print(y)
    print(len(layer_outputs))
    bins = 30
    in_bins= binning(layer_outputs, layers, epochs, bins)
    print(len(in_bins))


    mi = mutualInformation(y,x)
    print('MI:',mi)
    print('=============================================================')
    print('ENCODER:')
    I_XT = information_flow(x_train, in_bins, layers, epochs)
    print('=============================================================')
    print('DECODER:')
    I_YT = information_flow(y_train, in_bins, layers, epochs)

    #info_plane(I_XT,I_YT)

    information_plane2(I_XT, I_YT, epochs, 1, mi)
    plot_epochs(I_XT, I_YT)
    plot_for_epoch(I_XT, I_YT, epochs, 1, mi)

    print("Main ende ====")
