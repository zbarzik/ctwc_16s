#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import numpy as np

INITIALIZED = False

def plot_mat(mat, xlabel=None, ylabel=None, header=None):
    if not INITIALIZED:
        return

    plt.matshow(mat)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if header is not None:
        plt.title(header)
    plt.draw()
    plt.pause(0.001)

def show_plots():
    plt.show()
    raw_input("Press [Enter] to close all plots...")

def init():
    try:
        import matplotlib.pyplot as plt
        plt.ion()
        plt.set_cmap('hot')
        INITIALIZED = True
    except Exception:
        INITIALIZED = False

def wait_for_user():
    if INITIALIZED:
        raw_input("Press Enter to continue...")

def test():
    init()
    mat = np.random.rand(1200,18000)
    plot_mat(mat, str(18000), str(1200), 'large rectangular matrix')
    mat2 = np.random.rand(300, 300)
    plot_mat(mat2, header="header")
    show_plots()

if __name__ == '__main__':
    test()
