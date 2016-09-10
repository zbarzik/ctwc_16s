#!/usr/bin/python
from common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP
from multiprocessing import Pool
import warnings
import test_data

import numpy as np
import math

REAL_DATA = True

SAMPLE_THRESHOLD = 2
USE_LOG_XFORM = True
WEIGHTED_UNIFRAC = False
NUM_THREADS = 16
GENERATE_FILE_COLS = True
GENERATE_FILE_ROWS = True
COL_DISTANCE_MATRIX_FILE = './sample_distance.dat'
ROW_DISTANCE_MATRIX_FILE = './bacteria_distance.dat'
UNIFRAC_DIST_FILE = './unifrac_dist_mat.pkl'
UNIFRAC_HASH_FILE = './unifrac_dist_mat.hash'
SQUARE_UNIFRAC_DISTANCE = False


def __unifrac_prepare_entry_for_dictionary(args):
    data, otu_ind, otu, otus, otu_filter, samples, sample_filter = args
    samp_dict = {}
    for samp_ind, samp in enumerate(samples):
        if samp in sample_filter or otu in otu_filter:
            samp_dict[samp] = 0
        else:
            if USE_LOG_XFORM:
                samp_dict[samp] = 0 if data[samp_ind, otu_ind] < SAMPLE_THRESHOLD else np.log2(data[samp_ind, otu_ind])
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
    for samp_ind, samp in enumerate(samples):
        if samp not in sample_filter:
            uf_ind = uf_samples.index(samp)
            for other_ind, other_samp in enumerate(samples):
                uf_other_ind = uf_samples.index(other_samp)
                z[samp_ind, other_ind] = uf_dist_mat[uf_ind, uf_other_ind]
    return z

def __get_precalculated_unifrac_file_if_exists():
    import pickle
    try:
        with open(UNIFRAC_DIST_FILE, 'rb') as fn:
            try:
                mat = pickle.load(fn)
                return mat
            except Exception as e:
                WARN("Got an exception trying to read Unifrac distance matrix:\n" + str(e))
    except IOError:
        DEBUG("Couldn't open pre-calculated Unifrac distance matrix")
    return None

def __calculate_hash_for_data(data, sample_filter, otu_filter):
    return hash(hash(data.tostring()) + hash(str(sample_filter)) + hash(str(otu_filter))) # eh close enough

def __get_precalculated_unifrac_file_if_exists_for_data(data, sample_filter, otu_filter):
    h = __calculate_hash_for_data(data, sample_filter, otu_filter)
    try:
        with open(UNIFRAC_HASH_FILE, 'rb') as fn:
            hash_in_file = fn.readline()
            try:
                if long(hash_in_file) == h:
                    return __get_precalculated_unifrac_file_if_exists()
            except Exception as e:
                WARN("Got an exception trying to read Unifrac hash:\n" + str(e))
    except IOError:
        DEBUG("Couldn't open pre-calculated Unifrac hash")
    return None

def __save_calculated_unifrac_file_and_hash_for_data(data, sample_filter, otu_filter, mat):
    DEBUG("Saving calculated Unifrac distance matrix to file...")
    with open(UNIFRAC_HASH_FILE,'wb+') as fn:
        h = __calculate_hash_for_data(data, sample_filter, otu_filter)
        fn.write(str(h))
    with open(UNIFRAC_DIST_FILE, 'wb+') as fn:
        import pickle
        pickle.dump(mat, fn)

def unifrac_distance_rows(data, samples_arg=None, otus_arg=None, tree_arg=None, sample_filter=None, otu_filter=None):

    DEBUG("Starting unifrac_distance_rows...")
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
    return np.nan_to_num(dis)

def pearson_distance_rows(data, samples, otus, sample_filter, otu_filter):
    DEBUG("Starting pearson_distance_rows...")
    data = np.copy(data)
    DEBUG("Filtering Samples...")
    try:
        if samples is not list:
            samples = samples.tolist()
        cols_filter = [ samples.index(samp) for samp in ( sample_filter if sample_filter else [] ) ]
    except ValueError as e:
        FATAL("Trying to filter out non-existing samples: {0}".format(str(e)))
    for col in cols_filter:
        data[:, col] = 0
    DEBUG("Filtering OTUs...")
    try:
        if otus is not list:
            otus = otus.tolist()
        rows_filter = [ otus.index(otu) for otu in ( otu_filter if otu_filter else [] ) ]
    except ValueError as e:
        FATAL("Trying to filter out non-existing OTUs: {0}".format(str(e)))
    for row in rows_filter:
        data[row, :] = 0

    DEBUG("Log transform OTU abundance...")
    data[data < SAMPLE_THRESHOLD] = 0.0
    data = np.ma.log2(data)

    DEBUG("Calculating correlation...")
    correlation = np.corrcoef(data)
    DEBUG("Calculating dissimilarity...")
    res = dissimilarity_from_correlation(correlation)
    DEBUG("Finished calculating OTU distance matrix.")
    return res

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

def get_output_filename_by_type(mat_type):
    if mat_type == 'col':
        return COL_DISTANCE_MATRIX_FILE
    elif mat_type == 'row':
        return ROW_DISTANCE_MATRIX_FILE
    else:
        FATAL('Unknown matrix type')

def generate_spc_output_files(dist_mat, output_filename):
    with open(output_filename, 'w+') as fn:
        for r, _ in enumerate(dist_mat):
            for c in range(r):
                fn.write("{0} {1} {2}\n".format(r, c, dist_mat[r][c]))
    return r

def test():
    samples, otus, tree, data = get_data(REAL_DATA)
    _, cols_dist = get_distance_matrices(data, tree, samples, otus, skip_rows=True)
    rows_dist, _ = get_distance_matrices(data, tree, samples, otus, skip_cols=True)
    if GENERATE_FILE_COLS:
        generate_spc_output_files(cols_dist, get_output_filename_by_type('col'))
    if GENERATE_FILE_ROWS:
        generate_spc_output_files(rows_dist, get_output_filename_by_type('row'))


if __name__ == '__main__':
    test()
