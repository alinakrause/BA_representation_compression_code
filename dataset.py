import numpy as np
from sklearn.model_selection import train_test_split

def loadData():
   # print('here')
    loaded_inputs = np.loadtxt('Xdata4.csv', delimiter=',')
    loaded_labels = np.loadtxt('Ydata4.csv', delimiter=',')
    x = loaded_inputs
    y = loaded_labels

    return x,y

def prepData(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
    #print(x_train)
    #print(y_train)
    #input_shape = len(x_train[0])
    #print(x_train.shape)
    #print(y_train.shape)
    return x_train, x_test, y_train, y_test