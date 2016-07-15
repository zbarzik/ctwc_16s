#!/usr/bin/python
from common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP
import warnings
import test_data

import numpy as np
import math

REAL_DATA = True

def __unifrac_prepare_dictionary_from_matrix_rows(data, samples, otus, sample_filter, otu_filter):
    num_samples, num_otus_in_sample = data.shape
    if num_samples != len(samples):
        FATAL("Number of sample lables {0} does not match number of samples {1}".format(num_samples, len(samples)))
    if num_otus_in_sample != len(otus):
        FATAL("Number of otus labels {0} does not match number of samples {1}".format(num_otus_in_sample, len(otus)))
    full_dict = {}
    for otu_ind, otu in enumerate(otus):
        samp_dict = {}
        for samp_ind, samp in enumerate(samples):
            if samp in sample_filter or otu in otu_filter:
                samp_dict[samp] = 0
            else:
                samp_dict[samp] = data[samp_ind, otu_ind]
        full_dict[otu] = samp_dict
    DEBUG("Full dictionary: {0}".format(full_dict))
    return full_dict

def __reorder_unifrac_distance_matrix_by_original_samples(unifrac_output, samples, sample_filter, otu_filter):
    uf_dist_mat = unifrac_output[0]
    uf_samples = unifrac_output[1]
    z = np.zeros((len(samples), len(samples)))
    for samp_ind, samp in enumerate(samples):
        if samp not in sample_filter:
            uf_ind = uf_samples.index(samp)
            for other_ind, other_samp in enumerate(samples):
                uf_other_ind = uf_samples.index(other_samp)
                z[samp_ind, other_ind] = uf_dist_mat[uf_ind, uf_other_ind]
    return z

def unifrac_distance_rows(data, samples_arg=None, otus_arg=None, tree_arg=None, sample_filter=None, otu_filter=None):
    if sample_filter == None:
        sample_filter = []
    if otu_filter == None:
        otu_filter = []
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

    data_dict = __unifrac_prepare_dictionary_from_matrix_rows(data, samples, otus, sample_filter, otu_filter)
    unifrac = fast_unifrac(tree, data_dict, weighted=True)
    DEBUG("Unifrac results: {0}".format(unifrac))
    mat = __reorder_unifrac_distance_matrix_by_original_samples(unifrac['distance_matrix'], samples, sample_filter, otu_filter)
    found_nans = set([])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isinf(mat[i][j]) or math.isnan(mat[i][j]):
                found_nans.add(str(mat[i][j]))
                mat[i][j] = 0.0
    if len(found_nans) > 0:
        WARN("Found {0} value(s) in unifrac matrix".format("".join(str(x) for x in found_nans)))
    return mat

def unifrac_distance_cols(data, samples_arg=None, otus_arg=None, tree_arg=None, sample_filter=None, otu_filter=None):
    return unifrac_distance_rows(data.transpose(), samples_arg, otus_arg, tree_arg, sample_filter, otu_filter)

def dissimilarity_from_correlation(correlation):
    ones = np.ones(correlation.shape)
    return ones - abs(correlation)

def pearson_distance_rows(data, samples, otus, sample_filter, otu_filter):
    try:
        if samples is not list:
            samples = samples.tolist()
        cols_filter = [ samples.index(samp) for samp in ( sample_filter if sample_filter else [] ) ]
    except ValueError as e:
        FATAL("Trying to filter out non-existing samples: {0}".format(str(e)))
    try:
        if otus is not list:
            otus = otus.tolist()
        rows_filter = [ otus.index(otu) for otu in ( otu_filter if otu_filter else [] ) ]
    except ValueError as e:
        FATAL("Trying to filter out non-existing OTUs: {0}".format(str(e)))

    data = np.copy(data)
    for col in cols_filter:
        DEBUG("Data before: {0}".format(data))
        DEBUG("Clearing column {0}".format(col))
        data[:, col] = 0
        DEBUG("Data after: {0}".format(data))
    for row in rows_filter:
        DEBUG("Data before: {0}".format(data))
        DEBUG("Clearing row {0}".format(row))
        data[row, :] = 0
        DEBUG("Data after: {0}".format(data))

    correlation = np.corrcoef(data)
    for i in range(correlation.shape[0]):
        for j in range(correlation.shape[1]):
            if correlation[i][j] != correlation[i][j]:
                correlation[i][j] = 0
    return dissimilarity_from_correlation(correlation)

def pearson_distance_cols(data, samples, otus, sample_filter, otu_filter):
    return pearson_distance_rows(data.transpose(), samples, otus, sample_filter, otu_filter)

def euclidean_distance_rows(data):
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

def get_distance_matrices(data, tree, samples, otus, sample_filter=None, otu_filter=None, skip_cols=False, skip_rows=False):
    cols_dist = None
    rows_dist = None
    if not skip_cols:
        cols_dist = unifrac_distance_cols(data=data, samples_arg=samples, otus_arg=otus, tree_arg=tree, sample_filter=sample_filter, otu_filter=otu_filter)
    if not skip_rows:
        rows_dist = pearson_distance_rows(data, samples, otus, sample_filter, otu_filter)
    return rows_dist, cols_dist

def get_data(use_real_data):
    if use_real_data:
        INFO("Using real data")
        data, otus, samples = test_data.get_sample_biom_table()
        tree = test_data.get_gg_97_otu_tree()
    else:
        INFO("Using synthetic data")
        samples = test_data.get_default_samples()
        otus = test_data.get_default_otus()
        tree = test_data.get_default_tree(otus)
        data = test_data.get_default_data(otus, samples)
    return samples, otus, tree, data

def check_line(dist_matrix, index):
    l = len(dist_matrix[index].tolist())
    vec = dist_matrix[index].tolist()
    sum1 = 0
    sum2 = 0
    for i in range(len(vec) / 2):
        sum1 += vec[i]
        sum2 += vec[len(vec) / 2 + i]
    return sum1 < sum2

def test():
    samples, otus, tree, data = get_data(REAL_DATA)
    _, cols_dist = get_distance_matrices(data, tree, samples, otus, skip_rows=True)
    for i, col in enumerate(cols_dist):
        res = check_line(cols_dist, i)
        if res: INFO("{0} checks out".format(i))
        else: INFO("{0} doesn't check out".format(i))

if __name__ == '__main__':
    test()
