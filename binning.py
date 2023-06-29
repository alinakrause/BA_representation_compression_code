import numpy as np

#print(layer_outputs)

#bins = 30


def binning(layer_outputs, layers, epochs, bins):
    for layer in range(len(layers)):
        for epoch in range(epochs):
            layer_activations = layer_outputs[epoch][layer][0]
            min = np.min(layer_activations)
            max = np.max(layer_activations)
            bin_size = np.linspace(min, max, bins)
            discretized_activations = np.digitize(layer_activations, bin_size)
            layer_outputs[epoch][layer][0] = discretized_activations

    return layer_outputs

# Discretize the continous output of the layers

#activations_list = discretization(activations_list,bins, layers, epochs)
#in_bins= binning(layer_outputs, layers, epochs, bins)
