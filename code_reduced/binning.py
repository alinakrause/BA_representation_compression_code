"""
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import normalize


def mean(arr):
    """
    returns the mean on an array
    :param arr: array for that the mean needs to be calculated
    :return:
    """
    mean = sum(arr) / len(arr)
    return mean


def largest_bin_size(layer_outputs, layers, epochs, bins, a):
    """
    returns the maximal bin size for a distribution
    :param layer_outputs: layer output values
    :param layers: number of layers
    :param epochs: number of epochs
    :param bins: number of bins
    :param a: activation fucntion
    :return: mimimum, maximum, range of bins, bin size
    """
    if a == 'relu':
        ges_min = 100
        ges_max = -1
        for layer in range(len(layers)):
            for epoch in range(epochs):
                layer_activations = layer_outputs[epoch][layer][0]
                i_min = np.min(layer_activations)
                if (i_min < ges_min):
                    ges_min = i_min
                i_max = np.max(layer_activations)
                if (i_max > ges_max):
                    ges_max = i_max
    elif a == 'tanh':
        ges_min = -1
        ges_max = 1
    bin_size, size = np.linspace(ges_min, ges_max, bins, retstep=True)
    return ges_min, ges_max,bin_size,size


def total_max(layer_outputs, layers, epochs):
    """
    returns the total maximum of an array
    :param layer_outputs: layer output values
    :param layers: number of layers
    :param epochs: number of epochs
    :return: maxiimum value of an array
    """
    ges_min = 100
    ges_max = -1
    for layer in range(len(layers)):
        for epoch in range(epochs):
            layer_activations = layer_outputs[epoch][layer][0]
            i_min = np.min(layer_activations)
            if (i_min < ges_min):
                ges_min = i_min
            i_max = np.max(layer_activations)
            if (i_max > ges_max):
                ges_max = i_max

    return ges_max


def to_range(arr, new_min, new_max,mode):
    """
    shifts data to a different range for plots
    :param arr: data
    :param new_min: minimum of data
    :param new_max: maximum of data
    :param mode: mode
    :return:
    """
    new_max = new_max - new_min - (0.008 * new_max)
    new_min = 0+(0.008 * new_max)
    if (new_max < 0):
        new_max = 0
    if (new_min < 0):
        new_min = 0

    old_min = np.min(arr)
    old_max = np.max(arr)
    normalized_arr = (arr - old_min) / (old_max - old_min)
    reshaped_arr = normalized_arr * (new_max - new_min) + new_min

    return reshaped_arr




def binning_reduced(layer_outputs_orig, layers, epochs, bin_n,mode,activation):
    """
    equal distance binning
    :param layer_outputs_orig: layer output values
    :param layers: number of layers
    :param epochs: number of epochs
    :param bin_n: number of bins
    :param mode: mode
    :param activation: activation function
    :return: binned layer outputs
    """
    layer_outputs_binned = layer_outputs_orig

    for layer in range(len(layers)):
        for epoch in range(epochs):
            layer_activations = layer_outputs_orig[epoch][layer][0]

            i_min = np.min(layer_activations)
            i_max = np.max(layer_activations)
            if (mode == 'fixed'):
                if(activation == 'tanh'):
                    i_min = -1
                    i_max = 1
                elif(activation == 'relu'):
                    i_min = 0
                    i_max = total_max(layer_outputs_orig,layers,epochs)
            calc_bins, bin_size = np.linspace(i_min, i_max, bin_n, retstep=True)
            discretized_activations = np.digitize(layer_activations, calc_bins)


            layer_outputs_binned[epoch][layer][0] = discretized_activations

    return layer_outputs_binned



def binning(layer_outputs, layers, epochs, bins):
    """
    equal distance binning with visualizations to evaluate the results while coding
    ===============================================================================
    it generates plots of:
    histograms for a respective epoch
    the bin size
    the minimal and maximal layer outputs
    ===============================================================================
    :param layer_outputs: layer output values
    :param layers: number of layers
    :param epochs: number of epochs
    :param bins: bumber of bins
    :return: discretized activations
    """
    all_min_arr = []
    all_max_arr = []
    all_av_arr = []
    std_all = []

    for layer in range(len(layers)):
        min_arr = []
        max_arr = []
        av_arr = []
        std_arr = []
        bin_size_arr = []
        for epoch in range(epochs):
            layer_activations = layer_outputs[epoch][layer][0]

            i_min = np.min(layer_activations)
            i_max = np.max(layer_activations)

            av = sum(layer_activations) / len(layer_activations)
            std = np.std(layer_activations)
            av = mean(av)

            bin_size, size = np.linspace(i_min, i_max, bins,retstep=True)

            bin_size_arr.append(size)
            av_arr.append(av)
            std_arr.append(std)
            min_arr.append(i_min)
            max_arr.append(i_max)
            discretized_activations = np.digitize(layer_activations, bin_size)

            layer_outputs[epoch][layer][0] = discretized_activations
            if (epoch == 50):
                flat = (discretized_activations).flatten()

                out = pd.cut(flat, bins=bins, include_lowest=True)
                ax = out.value_counts().plot.bar(rot=0, color="b", figsize=(10, 6))
                ax.set_xticks([])
                ax.set_xticklabels([])
                plt.ylim(0,len(flat))
                plt.show()

        std_all.append(std_arr)
        all_min_arr.append(min_arr)
        all_max_arr.append(max_arr)
        all_av_arr.append(bin_size_arr)

    av = sum(bin_size_arr)/len(bin_size_arr)
    m_b = np.min(bin_size_arr)
    ma_b = np.max(bin_size_arr)
    print('average bin  size: ', av)
    print('minimum bin  size: ', m_b)
    print('maximum bin  size: ', ma_b)
    for layer in range(len(layers)):
        plt.plot(all_av_arr[layer], label='Layer {}'.format(str(layer+1)))
    plt.xlabel("Epochs")
    plt.ylabel("Bin Size")
    plt.legend()
    plt.show()
    for layer in range(len(layers)):
        plt.plot(all_max_arr[layer], label='Layer {}'.format(str(layer+1)))
    plt.xlabel("Epochs")
    plt.ylabel("Max Layer Outputs")
    plt.legend()
    plt.show()
    for layer in range(len(layers)):
        plt.plot(all_min_arr[layer], label='Layer {}'.format(str(layer+1)))
    plt.xlabel("Epochs")
    plt.ylabel("Min Layer Outputs")
    plt.legend()
    plt.show()

    return layer_outputs




