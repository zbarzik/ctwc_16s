#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP,save_to_file,load_from_file

import numpy as np

INITIALIZED = False
PLOT_RAW_FILE = './plot_raw-{0}.pklz'
PLOT_MAT_RAW_FILE = './plot_raw_mat-{0}.npz'

def plot_mat(mat, xlabel=None, ylabel=None, header=None):
    cameled_hdr = ''.join(x.title() for x in header.split())
    sanitized_hdr = ''.join(x for x in cameled_hdr if x.isalnum())
    save_to_file((xlabel, ylabel, header), PLOT_RAW_FILE.format(sanitized_hdr), mat, PLOT_MAT_RAW_FILE.format(sanitized_hdr))
    _plot_mat(mat, xlabel, ylabel, header)

def _plot_mat(mat, xlabel, ylabel, header):
    if not INITIALIZED:
        return

    import matplotlib.pyplot as plt

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
        globals()['INITIALIZED'] = True
    except Exception:
        globals()['INITIALIZED'] = False

def wait_for_user():
    if INITIALIZED:
        raw_input("Press Enter to continue...")

def plot_from_file(filename, mat_filename):
    pack = load_from_file(filename, True, mat_filename)
    if pack is None:
        return
    metadata, mat = pack
    xlabel, ylabel, header = metadata
    _plot_mat(mat, xlabel, ylabel, header)

def test():
    init()
    mat = np.random.rand(1200,18000)
    plot_mat(mat, str(18000), str(1200), 'large rectangular matrix')
    mat2 = np.random.rand(300, 300)
    plot_mat(mat2, header="header")
    show_plots()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        init()
        for desc in sys.argv[1:]:
            plot_from_file(PLOT_RAW_FILE.format(desc), PLOT_MAT_RAW_FILE.format(desc))
        wait_for_user()
        exit(0)

    test()
