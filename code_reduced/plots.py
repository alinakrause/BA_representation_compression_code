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

import matplotlib.pyplot as plt
import numpy as np



def information_plane2(IXT_array, ITY_array, num_epochs,binSize, m,act):
    """
    Generates information plane plot for all layer in one plot

    :param IXT_array: Mutual information layer output and input data
    :param ITY_array: Mutual information layer output and target data
    :param num_epochs: epochs of training
    ==================================================================
    only for naming:
    :param binSize: number of bins
    :param m: mode
    :param act: activation function
    :return:
    """
    IXT_list = []
    IYT_list = []
    max_index = len(IXT_array)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    a = np.linspace(0.8, 0, max_index)
    for i in range(0, max_index):

        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]

        IXT_list.append(IXT)
        IYT_list.append(ITY)

        plt.plot(IXT, ITY, label='layer {}'.format(str((i+1))), c=(a[i], a[i], a[i]))
        plt.scatter(IXT, ITY, marker='o', c=np.linspace(0, num_epochs, num_epochs), cmap='magma')
    plt.colorbar(label="Epochs", orientation="vertical")

    plt.legend(loc='lower right')
    #plt.savefig('All_{}_{}_{}bins_IP'.format(str(act),str(m),str(binSize)))
    plt.show()

def information_plane_single(IXT_array, ITY_array, num_epochs,binSize, m,act):
    """
    Generates information plane plot for every layer individually

    :param IXT_array: Mutual information layer output and input data
    :param ITY_array: Mutual information layer output and target data
    :param num_epochs: epochs of training
    ==================================================================
    only for naming:
    :param binSize: number of bins
    :param m: mode
    :param act: activation function
    :return:
    """
    IXT_list = []
    IYT_list = []
    max_index = len(IXT_array)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    a = np.linspace(1, 0, max_index)
    for i in range(0, max_index):

        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]

        IXT_list.append(IXT)
        IYT_list.append(ITY)

        plt.plot(IXT, ITY, label='layer {}'.format(str((i+1))), c=(a[i], a[i], a[i]))
        plt.scatter(IXT, ITY, marker='o', c=np.linspace(0, num_epochs, num_epochs), cmap='magma')
        plt.colorbar(label="Epochs", orientation="vertical")

        plt.legend()
        #plt.savefig('Indiv_{}_{}_{}bins_IP_layer{}'.format(str(act),str(m),str(binSize),str((i+1))))
        plt.close()
        #plt.show()


def MI_over_epochs(MI,layers,epochs,binSize,m,act):
    """
    Generates plot that shows MI as a function of epochs
    :param MI: mutual information data
    :param layers: number of layers
    :param epochs: number of epochs
    ==================================================================
    only for naming:
    :param binSize: number of bins
    :param m: mode
    :param act: activation function
    :return:
    """
    plt.figure()

    for i in range(0, len(layers)):
        plt.plot(np.arange(0, epochs, 1), MI[i], label='Layer {}'.format(str(i)))

    plt.xlabel("Epochs")
    plt.ylabel("I(T,X)")
    #plt.legend()
    #plt.savefig('MI_{}_{}_{}bins_MI'.format(str(act),str(m),str(binSize)))
   # plt.show()
    plt.close()




def plot_for_epoch(IXT_array, ITY_array, num_epochs):
    """
    plots MI behavior as scatterplot until a respective epoch
    :param IXT_array: Mutual information layer output and input data
    :param ITY_array: Mutual information layer output and target data
    :param num_epochs: epochs of training
    :return:
    """
    max_index = len(IXT_array)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    a = np.linspace(0, 1, max_index)
    epoch = num_epochs - 1
    for i in range(0, max_index):
        IXT = IXT_array[i, epoch]
        ITY = ITY_array[i, epoch]

        plt.xlim(0,12)
        plt.ylim(0,1)
        plt.plot(IXT, ITY, label='layer {}'.format(str(i)))
        plt.scatter(IXT, ITY, marker='o')

    plt.legend()
    plt.show()
