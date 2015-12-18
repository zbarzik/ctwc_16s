#!/usr/bin/python
from common import DEBUG,INFO,WARN,ERROR,FATAL

import numpy as np

def get_default_tree(bacteria):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    tree_str = "(((A:0.15)B:0.2,(C:0.3,D:0.4)E:0.6)F:0.1)G;"
    for i, bac in enumerate(bacteria):
        tree_str = tree_str.replace(chr(ord('A') + i), bac)
    tr = DndParser(tree_str, UniFracTreeNode)
    return tr

def get_default_samples():
    return ['mouth', 'butt', 'leg', 'armpit', 'foot']

def get_default_bacteria():
    return ['leonardo', 'donatello', 'raphael', 'michelangelo', 'splinter', 'shredder']

def get_default_data(bacteria, samples):
    num_bac = len(bacteria)
    num_samp = len(samples)
    data = np.zeros((num_bac, num_samp))
    for row, bac in enumerate(bacteria):
        for col, samp in enumerate(samples):
            data[row, col] = 1 if ord(samp[0]) < ord(bac[0]) else 0
    return data

def __unifrac_prepare_dictionary_from_matrix_rows(data, samples, bacteria):
    num_samples, num_bacteria_in_sample = data.shape
    if num_samples != len(samples):
        FATAL("Number of sample lables does not match number of samples")
    if num_bacteria_in_sample != len(bacteria):
        FATAL("Number of bacteria labels does not match number of samples")
    full_dict = {}
    for bac_ind, bac in enumerate(bacteria):
        samp_dict = {}
        for samp_ind, samp in enumerate(samples):
            samp_dict[samp] = data[samp_ind, bac_ind]
        full_dict[bac] = samp_dict
    return full_dict

def unifrac_distance_rows(data, samples_arg=None, bacteria_arg=None, tree_arg=None):
    from cogent.maths.unifrac.fast_unifrac import fast_unifrac

    if samples_arg==None:
        samples = get_default_samples()
    elif callable(samples_arg):
        samples = samples_arg()
    else:
        samples = samples_arg

    if bacteria_arg==None:
        bacteria = get_default_bacteria()
    elif callable(bacteria_arg):
        bacteria = bacteria_arg()
    else:
        bacteria = bacteria_arg

    if tree_arg==None:
        tree = get_default_tree(bacteria)
    elif callable(tree_arg):
        tree = tree_arg()
    else:
        tree = tree_arg

    data_dict = __unifrac_prepare_dictionary_from_matrix_rows(data, samples, bacteria)
    unifrac = fast_unifrac(tree, data_dict)
    return unifrac['distance_matrix']

def unifrac_distance_cols(data, samples_arg=None, bacteria_arg=None, tree_arg=None):
    return unifrac_distance_rows(data.transpose(), samples_arg, bacteria_arg, tree_arg)

def pearson_correlation_rows(data):
    return np.corrcoef(data)

def pearson_correlation_cols(data):
    return pearson_correlation_rows(data.transpose())

def euclideane_distance_rows(data):
    def _dist(vec1, vec2):
        from math import sqrt
        s = 0
        for i in range(0, min(len(vec1), len(vec2))):
            s += sqrt((vec1[i]-vec2[i])*(vec1[i]-vec2[i]))
        return s
    num_rows, row_vec_len = data.shape
    output = np.zeros((num_rows, num_rows))
    for i in range(0, num_rows):
        for j in range(0, num_rows):
            output[i][j] = _dist(data[i,:], data[j,:])
    return output

def euclidean_distance_cols(data):
    return euclidean_distance_rows(data.transpose())

if __name__ == '__main__':
    samples = get_default_samples()
    bacteria = get_default_bacteria()
    tree = get_default_tree(bacteria)
    data = get_default_data(bacteria, samples)
    print tree.asciiArt()
    print samples
    print bacteria
    print data
    print unifrac_distance_cols(data, samples, bacteria, tree)
