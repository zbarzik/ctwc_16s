#!/usr/bin/python
from ctwc__common import *

import ctwc__distance_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot, ctwc
import numpy as np

def test():
    np.seterr(all="ignore")
    samples, otus, tree, data, table = ctwc__distance_matrix.get_data(use_real_data=True, full_set=False)
    data = np.random.rand(data.shape[0], data.shape[1])
    test_otu_distance_matrix(samples, otus, tree, data, table)
    test_sample_distance_matrix(samples, otus, tree, data, table)

def test_otu_distance_matrix(samples, otus, tree, data, table):
    INFO("Test OTU distance matrix and filtering...")
    data[::2, :] = 30.0
    indices = range(36, data.shape[0])
    otu_filter_1, _ = ctwc.__prepare_otu_filters_from_indices(indices,
                                                              otus)
    indices = range(36, 196)
    otu_filter, _ = ctwc.__prepare_otu_filters_from_indices(indices,
                                                            otus,
                                                            otu_filter_1)

    rows_dist, _ = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               otu_filter=otu_filter,
                                                               skip_cols=True)

    INFO("Test OTU clustering and filtering...")
    picked_indices, _, _, _, _, _ = ctwc__cluster_rank.filter_rows_by_top_rank(data,
                                                                               rows_dist,
                                                                               otus)

    for ind in picked_indices:
        ASSERT(ind >= 36 and ind < 196)
        ASSERT(ind % 2 == 0)

    INFO("Passed OTU distance matrix and filtering tests")


def test_sample_distance_matrix(samples, otus, tree, data, table):
    INFO("Test sample distance matrix and filtering...")
    data[:,::2] = 50
    indices = range(36, data.shape[1])
    samp_filter_1, _ = ctwc.__prepare_sample_filters_from_indices(indices,
                                                                  samples)
    indices = range(36, 196)
    sample_filter, _ = ctwc.__prepare_sample_filters_from_indices(indices,
                                                                  samples,
                                                                  samp_filter_1)

    _, cols_dist = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               sample_filter=sample_filter,
                                                               skip_rows=True)


    INFO("Test sample clustering...")
    picked_indices, _, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data,
                                                                               cols_dist,
                                                                               samples)
    for ind in picked_indices:
        ASSERT(ind >= 36 and ind < 196)
        ASSERT(ind % 2 == 0)

    INFO("Passed sample distance matrix and filtering tests")

if __name__ == "__main__":
    test()
