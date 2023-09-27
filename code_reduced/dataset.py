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
from sklearn.model_selection import train_test_split

def loadData():
    """
    loads data from files
    :return:
    """
    loaded_inputs = np.loadtxt('Xdata5.csv', delimiter=',')
    loaded_labels = np.loadtxt('Ydata5.csv', delimiter=',')
    x = loaded_inputs
    y = loaded_labels

    return x,y

def prepData(x,y):
    """
    splits data into train and test data
    :param x: input data
    :param y: target data
    :return: splitted data
    """
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test