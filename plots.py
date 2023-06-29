import matplotlib.pyplot as plt
import numpy as np
def info_plane(I_XT,I_YT):
    plt.scatter(I_XT, I_YT, c = I_XT)
    plt.show()


def information_plane2(IXT_array, ITY_array, num_epochs, every_n, I_XY):

    max_index = len(IXT_array)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    a = np.linspace(0, 1, max_index)
    for i in range(0, max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]

        # plt.plot(IXT,ITY,label='layer {}'.format(str(i)), c = ((1/(i+1)),(1/(i+1)),(1/(i+1))))
        plt.plot(IXT, ITY, label='layer {}'.format(str(i)), c=(a[i], a[i], a[i]))
        plt.scatter(IXT, ITY, marker='o', c=np.linspace(0, num_epochs, num_epochs), cmap='magma')
    plt.colorbar(label="Epochs", orientation="vertical")

    plt.legend()
    plt.show()

def plot_epochs(I_XT,I_YT):
    # for i in range(0, 10):
    i = 149
    IXT = I_XT[:, i]
    ITY = I_YT[:, i]

    plt.scatter(IXT, ITY)
    # plt.colorbar(label="Epochs", orientation="horizontal")
    plt.show()

    # scatterplot pro layer ein punkt -> verschiedene farben?
    # über mehrere versuche für versch. epochen average plotten

def plot_for_epoch(IXT_array, ITY_array, num_epochs, every_n, I_XY):
    max_index = len(IXT_array)
    plt.xlabel('I(X;T)')
    plt.ylabel('I(T;Y)')
    a = np.linspace(0, 1, max_index)
    epoch = 149
    for i in range(0, max_index):
        IXT = IXT_array[i, epoch]
        ITY = ITY_array[i, epoch]

        # plt.plot(IXT,ITY,label='layer {}'.format(str(i)), c = ((1/(i+1)),(1/(i+1)),(1/(i+1))))
        plt.xlim(0,12)
        plt.ylim(0,1)
        plt.plot(IXT, ITY, label='layer {}'.format(str(i)))
        plt.scatter(IXT, ITY, marker='o')

    plt.legend()
    plt.show()
