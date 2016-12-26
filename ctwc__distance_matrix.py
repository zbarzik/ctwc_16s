#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP,save_to_file,load_from_file,has_value
from multiprocessing import Pool
import warnings
import ctwc__data_handler

import numpy as np
import math

REAL_DATA = True

SAMPLE_THRESHOLD = 2
USE_LOG_XFORM = True
WEIGHTED_UNIFRAC = False
NUM_THREADS = 32
COL_DISTANCE_MATRIX_FILE = './sample_distance.dat'
ROW_DISTANCE_MATRIX_FILE = './bacteria_distance.dat'
UNIFRAC_DIST_FILE = './unifrac_dist_mat-{0}.pklz'
SQUARE_UNIFRAC_DISTANCE = False
INF_VALUE = 1000
ALLOW_CACHING = False

def __unifrac_prepare_entry_for_dictionary(args):
    data, otu_ind, otu, otus, otu_filter, samples, sample_filter = args
    samp_dict = {}
    for samp_ind, samp in enumerate(samples):
        if ((sample_filter is not None and not has_value(sample_filter, samp))
             or
            (otu_filter is not None and not has_value(otu_filter, otu))):
            continue
        if USE_LOG_XFORM:
            samp_dict[samp] = 0 if data[samp_ind, otu_ind] < SAMPLE_THRESHOLD else np.log2(data[samp_ind, otu_ind])
        else:
            samp_dict[samp] = data[samp_ind, otu_ind]
    return {otu:samp_dict}

def __unifrac_prepare_dictionary_from_matrix_rows(data, samples, otus, sample_filter, otu_filter):
    num_samples, num_otus_in_sample = data.shape
    if num_samples != len(samples):
        FATAL("Number of sample lables {0} does not match number of samples {1}".format(num_samples, len(samples)))
    if num_otus_in_sample != len(otus):
        FATAL("Number of otus labels {0} does not match number of samples {1}".format(num_otus_in_sample, len(otus)))
    full_dict = {}
    args = []
    for otu_ind, otu in enumerate(otus):
        args.append((data, otu_ind, otu, otus, otu_filter, samples, sample_filter))
    p = Pool(NUM_THREADS)
    retvals = p.map(__unifrac_prepare_entry_for_dictionary, args)
    for retVal in retvals:
        full_dict.update(retVal)
    p.terminate()
    return full_dict

def __reorder_unifrac_distance_matrix_by_original_samples(unifrac_output, samples, sample_filter, otu_filter):
    uf_dist_mat = unifrac_output[0]
    uf_samples = unifrac_output[1]
    z = np.zeros((len(samples), len(samples)))
    z[:,:] = INF_VALUE
    np.fill_diagonal(z, 0.0)
    for samp_ind, samp in enumerate(samples):
        if sample_filter is None or has_value(sample_filter, samp):
            uf_ind = uf_samples.index(samp)
            for other_ind, other_samp in enumerate(samples):
                if sample_filter is None or has_value(sample_filter, other_samp):
                    uf_other_ind = uf_samples.index(other_samp)
                    z[samp_ind, other_ind] = uf_dist_mat[uf_ind, uf_other_ind]
    return z

def __get_precalculated_unifrac_file_if_exists(h):
    return load_from_file(UNIFRAC_DIST_FILE.format(h))

def __calculate_hash_for_data(data, sample_filter, otu_filter):
    return hash(str([ hash(data.tostring()), hash(str(sample_filter)), hash(str(otu_filter)) ]) ) # eh close enough

def __get_precalculated_unifrac_file_if_exists_for_data(data, sample_filter, otu_filter):
    h = __calculate_hash_for_data(data, sample_filter, otu_filter)
    return __get_precalculated_unifrac_file_if_exists(h)

def __save_calculated_unifrac_file_and_hash_for_data(data, sample_filter, otu_filter, mat):
    if not ALLOW_CACHING:
        return
    DEBUG("Saving calculated Unifrac distance matrix to file...")
    h = __calculate_hash_for_data(data, sample_filter, otu_filter)
    save_to_file(mat, UNIFRAC_DIST_FILE.format(h))

def __get_mask_from_filter(mat, filt):
    if filt is None:
        return np.zeros(mat.shape, dtype=bool)
    tmp_mask = np.ones(mat.shape[0], dtype=bool)
    tmp_mask[filt] = False
    mask = np.zeros(mat.shape, dtype=bool)
    mask[ np.nonzero(tmp_mask == True) ] = True
    mask[ : , np.nonzero(tmp_mask == True) ] = True
    return mask

def __increase_distance_for_filtered_samples(mat, filt):
    D = INF_VALUE # float('Inf')
    mask = __get_mask_from_filter(mat, filt)
    mat[ mask ] = D
    np.fill_diagonal(mat, 0.0)
    return mat

def unifrac_distance_rows(data, samples_arg=None, otus_arg=None, tree_arg=None, sample_filter=None, otu_filter=None):
    DEBUG("Starting unifrac_distance_rows...")
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

    mat = __get_precalculated_unifrac_file_if_exists_for_data(data, sample_filter, otu_filter)
    if mat is not None:
        DEBUG("Found previously calculated Unifrac data")
        return mat

    DEBUG("Preparing data dictionary...")
    data_dict = __unifrac_prepare_dictionary_from_matrix_rows(data, samples, otus, sample_filter, otu_filter)
    DEBUG("Running fast_unifrac...")
    unifrac = fast_unifrac(tree, data_dict, weighted=WEIGHTED_UNIFRAC)
    DEBUG("Unifrac results: {0}".format(unifrac))
    DEBUG("Reordering results...")
    mat = __reorder_unifrac_distance_matrix_by_original_samples(unifrac['distance_matrix'], samples, sample_filter, otu_filter)

    DEBUG("Setting distances for filtered items to large values...")
    if sample_filter is not None:
        filter_indices = [ ind for ind, samp in enumerate(samples) if has_value(sample_filter, samp) ]
        mat = __increase_distance_for_filtered_samples(mat, filter_indices)

    DEBUG("Fixing NaN/inf values...")
    mat = np.nan_to_num(mat)

    if SQUARE_UNIFRAC_DISTANCE:
        mat = np.multiply(mat, mat)

    __save_calculated_unifrac_file_and_hash_for_data(data, sample_filter, otu_filter, mat)
    DEBUG("Finished calculating Samples distance matrix.")
    return mat

def unifrac_distance_cols(data, samples_arg=None, otus_arg=None, tree_arg=None, sample_filter=None, otu_filter=None):
    return unifrac_distance_rows(data.transpose(), samples_arg, otus_arg, tree_arg, sample_filter, otu_filter)

def dissimilarity_from_correlation(correlation):
    ones = np.ones(correlation.shape)
    dis = ones - abs(correlation)
    dis = np.nan_to_num(dis)
    np.fill_diagonal(dis, 0.0)
    return dis

def pearson_distance_rows(data, samples, otus, sample_filter, otu_filter):
    return __calculate_otu_distance_rows(data, samples, otus, sample_filter, otu_filter, 'pearson')

def jaccard_distance_rows(data, samples, otus, sample_filter, otu_filter):
    return __calculate_otu_distance_rows(data, samples, otus, sample_filter, otu_filter, 'jaccard')

def __calculate_otu_distance_rows(data_in, samples, otus, sample_filter, otu_filter, metric):
    DEBUG("Starting distance calculation using {0} as a metric...".format(metric))
    data = np.copy(data_in)

    DEBUG("Filtering Samples...")
    try:
        if samples is not list:
            samples = samples.tolist()
        if sample_filter is not None:
            cols_filter = [ samples.index(samp) for has_value(sample_filter, samp) ]
        else:
            cols_filter = None
    except ValueError as e:
        FATAL("Trying to filter out non-existing samples: {0}".format(str(e)))

    if cols_filter is not None:
        mask = np.ones(data.shape, dtype=bool)
        mask[ :, cols_filter ] = False
        data[mask] = 0

    DEBUG("Filtering OTUs...")
    try:
        if otus is not list:
            otus = otus.tolist()
        if otu_filter is not None:
            rows_filter = [ otus.index(otu) for otu in otu_filter ]
        else:
            rows_filter = None
    except ValueError as e:
        FATAL("Trying to filter out non-existing OTUs: {0}".format(str(e)))

    if rows_filter is not None:
        mask = np.ones(data.shape, dtype=bool)
        mask[rows_filter] = False
        data[mask] = 0

    if USE_LOG_XFORM:
        DEBUG("Log transform OTU abundance...")
        data[data < SAMPLE_THRESHOLD] = 0.0
        data = np.ma.log2(data)
    if metric == 'pearson':
        res = __pearson_distance(data)
    elif metric == 'jaccard':
        res = __jaccard_distance(data)
    else:
        FATAL("Unknown metric requested: {0}".format(metric))

    mask = __get_mask_from_filter(res, rows_filter)
    res[mask] = INF_VALUE
    np.fill_diagonal(res, 0.0)

    DEBUG("Finished calculating OTU distance matrix.")
    return res

def __pearson_distance(data):
    DEBUG("Calculating Pearson correlation...")
    correlation = np.corrcoef(data)
    DEBUG("Calculating Pearson dissimilarity...")
    res = dissimilarity_from_correlation(correlation)
    return res

def __jaccard_distance(data):
    DEBUG("Calculating Jaccard Index...")
    bin_mat = np.zeros(data.shape)
    bin_mat[data > 0] = 1
    intersect_mat = np.dot(bin_mat, bin_mat.transpose()) # every cell is the number of common values, Jaccard nominator
    row_sums = intersect_mat.diagonal()
    union_mat = row_sums[:,None] + row_sums - intersect_mat
    with np.errstate(invalid='ignore'):
        jaccard_mat = np.divide(intersect_mat, union_mat)
    jaccard_mat[np.isnan(jaccard_mat)] = 0 # it's 1 by definition but we want to ignore zero vectors
    DEBUG("Calculating Jaccard dissimilarity...")
    res = dissimilarity_from_correlation(jaccard_mat)
    return res

def jaccard_distance_cols(data, samples, otus, sample_filter, otu_filter):
    return jaccard_distance_rows(data.transpose(), samples, otus, sample_filter, otu_filter)

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
        rows_dist = jaccard_distance_rows(data, samples, otus, sample_filter, otu_filter)
    return rows_dist, cols_dist

def get_data(use_real_data=True, full_set=True):
    if use_real_data:
        INFO("Using real data")
        data, otus, samples, table = ctwc__data_handler.get_sample_biom_table(full_set=full_set)
        tree = ctwc__data_handler.get_gg_97_otu_tree()
    else:
        INFO("Using synthetic data")
        samples = ctwc__data_handler.get_default_samples()
        otus = ctwc__data_handler.get_default_otus()
        tree = ctwc__data_handler.get_default_tree(otus)
        data = ctwc__data_handler.get_default_data(otus, samples)
        table = None
    return samples, otus, tree, data, table

def __get_output_filename_by_type(mat_type):
    if mat_type == 'col':
        return COL_DISTANCE_MATRIX_FILE
    elif mat_type == 'row':
        return ROW_DISTANCE_MATRIX_FILE
    else:
        FATAL('Unknown matrix type')

def test():
    samples, otus, tree, data, table = get_data(REAL_DATA)
    INFO("Calculating cols dist")
    _, cols_dist = get_distance_matrices(data, tree, samples, otus, skip_rows=True)
    INFO("Calculating rows dist")
    rows_dist, _ = get_distance_matrices(data, tree, samples, otus, skip_cols=True)
    #otu_filter = otus
    #sample_filter = samples
    #rows_dist, cols_dist = get_distance_matrices(data, tree, samples, otus, otu_filter=otu_filter, sample_filter=sample_filter)

if __name__ == '__main__':
    test()
