#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import matplotlib.pyplot as plt
import numpy as np

def plot_mat(mat, label=None):
    plt.matshow(mat)
    if label is not None:
        plt.ylabel(label)
    plt.draw()

def show_plots():
    plt.show()

def test():
    import matplotlib.pyplot as plt
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.show(block=False)
    plt.show()
    #mat = np.random.rand(1200,18000)
    #plot_mat(mat)


if __name__ == '__main__':
    test()
