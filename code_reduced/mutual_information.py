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
from scipy.stats import entropy


def join(a, b):
    """
    Joint two arrays along axis
    :param a: first data sample
    :param b: second data sample
    :return: joint data
    """
    if len(a.shape) == 1:
        a = a[:, None]
    if len(b.shape) == 1:
        b = b[:, None]

    if a.shape[0] != b.shape[0]:
        raise ValueError("Input arrays must have the same number of rows.")

    rows = a.shape[0]
    columns = a.shape[1] + b.shape[1]
    result = [[0] * columns for _ in range(rows)]

    for i in range(rows):
        for j in range(a.shape[1]):
            result[i][j] = a[i][j]
        for j in range(b.shape[1]):
            result[i][a.shape[1] + j] = b[i][j]

    return result


def entropy(A):
    """
    Calculate shannon entropy
    :param A: data
    :return:shannon entropy of data A
    """

    #determine probability by frequency of observed ssamples
    values, counts = np.unique(A, return_counts=True, axis=0)
    pA = counts/len(A)

    #calculate entropy
    e = -np.sum(pA * np.log2(pA))
    return e



def jointEntropy(A,B):
    """
    Calculate joint entropy for two data samples
    :param A: first data sample
    :param B: second data sample
    :return: joint entropy of data samples
    """

    # jointly observe both data samples
    joint_data = join(A, B)

    # calculate entropy for joint data
    joint_entropy = entropy(joint_data)
    return joint_entropy


def conditionalEntropy(A, B):
    """
    Calculate conditional entropy for two data samples
    :param A: first data sample
    :param B: second data sample
    :return: conditional entropy for data samples
    """
    entropy_ = entropy(B)
    joint_entropy = jointEntropy(A,B)
    # conditional entropy can be calculated by: H(A,B) − H(A)
    conditional_entropy = joint_entropy - entropy_
    return conditional_entropy



#Mutual Information
def mutualInformation(A, B):
    """
    Calculate mutual information for two data samples
    :param A: first data sample
    :param B: second data sample
    :return: mutual information for data samples
    """
    entropyA = entropy(A)
    conditional_entropy = conditionalEntropy(A,B)
    #MI can be calculated by: H(A) − H(A|B)
    mutual_information = entropyA - conditional_entropy

    return mutual_information


#test MI
def test_MI():
    loaded_inputs = np.loadtxt('Xdata5.csv', delimiter=',')
    loaded_labels = np.loadtxt('Ydata5.csv', delimiter=',')
    x = loaded_inputs
    y = loaded_labels
    I_XY = mutualInformation(y,x)
    return I_XY
