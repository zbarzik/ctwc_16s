#!/usr/bin/python
from ctwc__common import *

import numpy as np

INITIALIZED = False
PLOT_RAW_FILE = RESULTS_PATH+'plot_raw-{0}.pklz'
PLOT_MAT_RAW_FILE = RESULTS_PATH+'plot_raw_mat-{0}.npz'
PLOT_PNG_FILE = RESULTS_PATH+'plot-{0}.png'
SHOW_ON_SCREEN = False


def plot_mat(mat, xlabel=None, ylabel=None, header=None):
    sanitized_hdr = make_camel_from_string(header)
    save_to_file((xlabel, ylabel, header), PLOT_RAW_FILE.format(sanitized_hdr), mat, PLOT_MAT_RAW_FILE.format(sanitized_hdr))
    __plot_mat(mat, xlabel, ylabel, header)

def __plot_mat(mat, xlabel, ylabel, header):
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

    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=10)
    plt.xticks(rotation=45)
    plt.savefig(PLOT_PNG_FILE.format(make_camel_from_string(header)), dpi=1000)
    if SHOW_ON_SCREEN:
        plt.draw()
        plt.pause(0.001)


def show_plots():
    if SHOW_ON_SCREEN:
        plt.show()
        raw_input("Press [Enter] to close all plots...")

def init():
    try:
        import matplotlib
        matplotlib.use('Agg')
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
    __plot_mat(mat, xlabel, ylabel, header)

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
        if sys.argv[1] == "all":
            import glob
            for fn in glob.glob(PLOT_RAW_FILE.format("*")):
                tmplt = PLOT_RAW_FILE.format("@")
                desc = fn[len(templ.split("@")[0]):-len(tmplt.split("@")[1])]
                INFO(desc)
                plot_from_file(PLOT_RAW_FILE.format(desc), PLOT_MAT_RAW_FILE.format(desc))
        else:
            for desc in sys.argv[1:]:
                plot_from_file(PLOT_RAW_FILE.format(desc), PLOT_MAT_RAW_FILE.format(desc))
        wait_for_user()
        exit()
    else:
        test()
