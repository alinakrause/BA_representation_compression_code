from dataset import loadData, prepData
from network import build_model
from mutual_information import mutualInformation
from binning import binning
from information_flow import information_flow
from plots import information_plane2

layer_outputs = []


if __name__ == "__main__":
    x,y = loadData()
    x_train, x_test, y_train, y_test = prepData(x,y)
    shape = len(x_train[0])

    layer_outputs = build_model(x_train,x_test, y_train, y_test, shape, layer_outputs)

    bins = 30
    in_bins= binning(layer_outputs, layers, epochs, bins)
    print(len(in_bins))






    mi = mutualInformation(y,x)
    print('MI:',mi)
    print('=============================================================')
    print('ENCODER:')
    I_XT = information_flow(x_train, in_bins, layers, epochs)
    print('=============================================================')
    print('DECODER:')
    I_YT = information_flow(y_train, in_bins, layers, epochs)

    #info_plane(I_XT,I_YT)

    information_plane2(I_XT, I_YT, epochs, 1, mi)
    plot_epochs(I_XT, I_YT)

    print("Main ende ====")