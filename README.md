# BA_representation_compression_code
This code is developed for the bachelor thesis "Do Deep Neural Networks perfrom Representation Compression?"
The code in this repository trains a neural network with either tanh or relu activation function. The outputs of each layer are stored.
The layer outputs get discretized using an equal width binning method with either fixed or adapted bounds defining the range.
Finally, the mutual information is estimated for the binned layer outputs and the input data, as well as the target labels and plotted in an information plane plot.
The code is modified such that it completely runs for one trial. To run the code, run the network_all.py file.
