"""
This code trains a neural network with either tanh or relu activation function. The outputs of each layer are stored.
The layer outputs get discretized using an equal width binning method with either fixed or adapted bounds defining the range.
Finally, the mutual information is esttimated for the binned layer outputs and the input data, as well as the target labelös and plotted iin an information plane plot.

Copyright (c) 2023 Alina Krause

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.callbacks import LambdaCallback
from binning import binning, binning_reduced
from mutual_information import mutualInformation
from information_flow import information_flow
from plots import information_plane2, MI_over_epochs, information_plane_single
from dataset import loadData, prepData
from sklearn.preprocessing import normalize
import seaborn as sn
from sklearn.metrics import confusion_matrix

layer_outputs = []
weights = []



def get_output(model, x_train):
    """
    function to store the layer otputs and weights of a network
    :param model: the network
    :param x_train: input training data
    :return:
    """
    outputs = []
    functors = []
    layer_activations = []
    layer_weights = []

    # all output tensors from each layer in the model
    for layer in model.layers:
        outputs.append(layer.output)
        layer_weights.append(layer.get_weights())

    # evaluate the output tensor given an input
    for out in outputs:
        functor = K.function([model.input], [out])
        functors.append(functor)

    # extract layer activations for input data
    for f in functors:
        activation = f([x_train])
        layer_activations.append(activation)

    layer_outputs.append(layer_activations)
    weights.append(layer_weights)



if __name__ == "__main__":
    x, y = loadData()
    x_train, x_test, y_train, y_test = prepData(x, y)

    #the network architecture and training
    shape = len(x_train[0])
    layers = [12, 10, 7, 5, 4, 3, 2, 1]
    activationFuctions = ['tanh', 'relu']
    activation = activationFuctions[1]
    batch_size = 180
    epochs = 100

    model = keras.models.Sequential([
        keras.layers.Dense(layers[0], activation=activation, input_shape=(shape,)),
        keras.layers.Dense(layers[1], activation=activation),
        keras.layers.Dense(layers[2], activation=activation),
        keras.layers.Dense(layers[3], activation=activation),
        keras.layers.Dense(layers[4], activation=activation),
        keras.layers.Dense(layers[5], activation=activation),
        keras.layers.Dense(layers[6], activation=activation),
        keras.layers.Dense(layers[7], activation='sigmoid'),
    ])

    print(model.summary())

    loss = tf.keras.losses.BinaryCrossentropy()
    optim = tf.keras.optimizers.experimental.SGD(learning_rate=0.01)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    # at the end of every epoch
    callback = LambdaCallback(on_epoch_end=lambda batch, logs: get_output(model, x_train))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, callbacks=[callback])
    model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

    modes = ["adapted", "fixed"]
    mi = mutualInformation(y, x)
    outputs = layer_outputs
    mode = 1
    outputs = layer_outputs
    bins = 90
    # can be used to store data for every run
    # pd.DataFrame(outputs).to_csv('layerOutputs_{}_{}_{}bins'.format(str(activation),str(mode),str(bins)))

    # binning the data
    in_bins = binning_reduced(outputs, layers, epochs, bins, mode, activation)
    print('=============================================================')
    print('I(X;T) for layer:')
    I_XT = information_flow(x_train, in_bins, layers, epochs)
    print('=============================================================')
    print('I(T;Y) for layer:')
    I_YT = information_flow(y_train, in_bins, layers, epochs)

    # plot the results
    information_plane2(I_XT, I_YT, epochs, bins, modes[mode], activation)
    information_plane_single(I_XT, I_YT, epochs, bins, modes[mode], activation)
    MI_over_epochs(I_XT, layers, epochs, bins, modes[mode], activation)











