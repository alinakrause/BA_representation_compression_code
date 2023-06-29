import numpy as np
from mutual_information import mutualInformation



def information_flow(var,activations_list, layers, epochs):
    #epoch = 190
    num_layers = len(layers)
    #layer = 5
    I_flow = np.zeros((num_layers,epochs))

    for layer in range(0,num_layers):
        print(layer+1,'/ 8')
        for epoch in range(0,epochs):
            I_flow[layer,epoch] = mutualInformation(activations_list[epoch][layer][0],var)

    return I_flow
