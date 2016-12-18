#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import ctwc__distance_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot, ctwc
import numpy as np

def test():
    np.seterr(all="ignore")
    samples, otus, tree, data, table = ctwc__distance_matrix.get_data(True)
    test_otu_distance_matrix(samples, otus, tree, data, table)
    test_sample_distance_matrix(samples, otus, tree, data, table)

def test_otu_distance_matrix(samples, otus, tree, data, table):
    INFO("Test OTU distance matrix and filtering...")
    data[::4, :] = 50
    indices = range(36)
    otu_filter_1, _ = ctwc.__prepare_otu_filters_from_indices(indices,
                                                           otus,
                                                           [])
    indices = range(36, 96)
    otu_filter, _ = ctwc.__prepare_otu_filters_from_indices(indices,
                                                         otus,
                                                         otu_filter_1)

    rows_dist, _ = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               otu_filter=otu_filter,
                                                               sample_filter=[],
                                                               skip_cols=True)

    INFO("Test filtering...")
    for row_ind, row in enumerate(rows_dist):
        count = len( [ cell for cell in row if cell == ctwc__distance_matrix.INF_VALUE ] )
        if row_ind < 96:
            ASSERT(count == data.shape[0] - 1)
        else:
            ASSERT(count == 96)

    INFO("Passed OTU distance matrix and filtering tests")
    #picked_indices, _, _, _, _, _ = 

def test_sample_distance_matrix(samples, otus, tree, data, table):
    INFO("Test sample distance matrix and filtering...")
    data[:,::15] = 50
    indices = range(0, 36)
    samp_filter_1, _ = ctwc.__prepare_sample_filters_from_indices(indices,
                                                               samples,
                                                               [])
    indices = range(36, 96)
    sample_filter, _ = ctwc.__prepare_sample_filters_from_indices(indices,
                                                               samples,
                                                               samp_filter_1)

    _, cols_dist = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               otu_filter=[],
                                                               sample_filter=sample_filter,
                                                               skip_rows=True)

    INFO("Test filtering...")
    for row_ind, row in enumerate(cols_dist):
        count = len( [ cell for cell in row if cell == ctwc__distance_matrix.INF_VALUE ] )
        if row_ind < 96:
            ASSERT(count == data.shape[1] - 1)
        else:
            ASSERT(count == 96)

    #INFO("Test clustering...")
    #picked_indices, _, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data,
    #                                                                           cols_dist,
    #                                                                           samples)
    #for ind in picked_indices:
    #    ASSERT(ind % 15 == 0)
    #    ASSERT(ind >= 96)

    #INFO("Passed sample distance matrix and filtering tests")

if __name__ == "__main__":
    test()
