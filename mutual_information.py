

##Entropy. ==> X und Y tauschen?
## vielleicht über kullback liebler divergence? oder entropy funcrtion nutzen

import numpy as np
from scipy.stats import entropy




def entropy(A):

    values, counts = np.unique(A, return_counts=True, axis=0)
    pA = counts/len(A)
    #print(unique)
    #print(counts)
    #en = np.sum((-1)*prob*np.log2(prob))
    #print(en)
    e = -np.sum(pA * np.log2(pA))
    #e_test = -np.sum(Y * np.log2(Y))
    return e

#Joint Entropy

def jointEntropy(A,B):

    joint_data = np.c_[A, B]
    joint_entropy = entropy(joint_data)
    return joint_entropy

#Conditional Entropy
def conditionalEntropy(A, B):

    entropy_ = entropy(B)
    joint_entropy = jointEntropy(A,B)
    conditional_entropy = joint_entropy - entropy_
    return conditional_entropy



#Mutual Information
def mutualInformation(A, B):

    #entropyX = entropy(B)
    entropyA = entropy(A)
    conditional_entropy = conditionalEntropy(A,B)
    mutual_information = entropyA - conditional_entropy

   # print((entropy(Y)-conditionalEntropy(Y,X)) == (entropy(Y)  - entropy(np.c_[Y,X]) - entropy(X)))
    return mutual_information


#test MI
loaded_inputs = np.loadtxt('Xdata4.csv', delimiter=',')
loaded_labels = np.loadtxt('Ydata4.csv', delimiter=',')
x = loaded_inputs
y = loaded_labels
I_XY = mutualInformation(y,x)
print(I_XY)
