#!/usr/bin/python
from common import DEBUG,INFO,WARN,ERROR,FATAL
import warnings

import numpy as np

def get_default_tree(otus):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    tree_str = "(((A:0.15)B:0.2,(C:0.3,D:0.4)E:0.6)F:0.1)G;"
    for i, otu in enumerate(otus):
        tree_str = tree_str.replace(chr(ord('A') + i), otu)
    tr = DndParser(tree_str, UniFracTreeNode)
    return tr

def get_tree_from_file(path):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    f = open(path, 'r')
    tr = DndParser(f.read(), UniFracTreeNode)
    return tr

def get_gg_97_otu_tree():
    tr = get_tree_from_file('97_otus.tree')
    return tr

def get_biom_table_from_file(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from biom import parse_table
    with open(path) as f:
        table = parse_table(f)
        return table
    return None

def get_sample_biom_table():
    table = get_biom_table_from_file('305_otu_table.json')
    if table is not None:
        return table.matrix_data.todense(), table.ids('observation'), table.ids('sample')
    return None

def get_default_samples():
    return ['mouth', 'butt', 'leg', 'armpit', 'foot']

def get_default_otus():
    return ['leonardo', 'donatello', 'raphael', 'michelangelo', 'splinter', 'shredder', 'april']

def get_default_data(otus, samples):
    num_otu = len(otus)
    num_samp = len(samples)
    data = np.zeros((num_otu, num_samp))
    for row, otu in enumerate(otus):
        for col, samp in enumerate(samples):
            data[row, col] = 1 if ord(samp[0]) < ord(otu[0]) else 0
    return data

def __unifrac_prepare_dictionary_from_matrix_rows(data, samples, otus):
    num_samples, num_otus_in_sample = data.shape
    if num_samples != len(samples):
        FATAL("Number of sample lables {0} does not match number of samples {1}".format(num_samples, len(samples)))
    if num_otus_in_sample != len(otus):
        FATAL("Number of otus labels {0} does not match number of samples {1}".format(num_otus_in_sample, len(otus)))
    full_dict = {}
    for otu_ind, otu in enumerate(otus):
        samp_dict = {}
        for samp_ind, samp in enumerate(samples):
            samp_dict[samp] = data[samp_ind, otu_ind]
        full_dict[otu] = samp_dict
    return full_dict

def unifrac_distance_rows(data, samples_arg=None, otus_arg=None, tree_arg=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from cogent.maths.unifrac.fast_unifrac import fast_unifrac
    if samples_arg is None:
        samples = get_default_samples()
    elif callable(samples_arg):
        samples = samples_arg()
    else:
        samples = samples_arg

    if otus_arg is None:
        otus = get_default_otus()
    elif callable(otus_arg):
        otus = otus_arg()
    else:
        otus = otus_arg

    if tree_arg is None:
        tree = get_default_tree(otus)
    elif callable(tree_arg):
        tree = tree_arg()
    else:
        tree = tree_arg

    data_dict = __unifrac_prepare_dictionary_from_matrix_rows(data, samples, otus)
    unifrac = fast_unifrac(tree, data_dict)
    return unifrac['distance_matrix']

def unifrac_distance_cols(data, samples_arg=None, otus_arg=None, tree_arg=None):
    return unifrac_distance_rows(data.transpose(), samples_arg, otus_arg, tree_arg)

def dissimilarity_from_correlation(correlation):
    ones = np.ones(correlation.shape)
    return ones - abs(correlation)

def pearson_distance_rows(data):
    correlation = np.corrcoef(data)
    return dissimilarity_from_correlation(correlation)

def pearson_distance_cols(data):
    return pearson_distance_rows(data.transpose())

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

def get_distance_matrices(data, samples=None, tree=None, otus=None):
    cols_dist = unifrac_distance_cols(data=data, samples_arg=samples, otus_arg=otus, tree_arg=tree)
    rows_dist = pearson_distance_rows(data)
    return rows_dist, cols_dist

def live_debug():
    table, otus, samples = get_sample_biom_table()
    import pdb
    pdb.set_trace()

if __name__ == '__main__':
    data, otus, samples = get_sample_biom_table()
    tree = get_gg_97_otu_tree()
    #samples = get_default_samples()
    #otus = get_default_otus()
    #tree = get_default_tree(otus)
    #data = get_default_data(otus, samples)
    rows_dist, cols_dist = get_distance_matrices(data, samples, tree, otus)
#    print "Tree:\n" + tree.asciiArt()
    print "Samples:\n" + str(samples)
    print "OTUs:\n" + str(otus)
#    print "Data:\n" + str(data)
    print "Rows Matrix:\n" + str(rows_dist)
    print "Cols Matrix:\n" + str(cols_dist)
