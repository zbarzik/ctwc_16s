#!/usr/bin/ipython
from common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import create_distance_matrix, rank_cluster
import numpy as np

LOG = INFO


def join_submatrices(mat_a, mat_b):
    if len(mat_b) == 0:
        return mat_a
    if len(mat_a) == 0:
        return mat_b
    axis = 1 if mat_a.shape[0] == mat_b.shape[0] else 0
    ret = np.concatenate((mat_a, mat_b), axis)
    ASSERT(ret.shape[axis] == mat_a.shape[axis] + mat_b.shape[axis])
    return np.squeeze(ret)

def ctwc_iteration(data, rows_dist, cols_dist):
    LOG("Now running on data size: {0}".format(data.shape))
    filtered_data, filtered_dist_matrix, picked_indices, last_rank, filtered_data_compliment, filtered_dist_matrix_compliment = rank_cluster.filter_rows_by_top_rank(
                                                                                                                                                     data, rows_dist)
    if len(filtered_data_compliment) == 0:
        LOG("Didn't filter out anything. last_rank = {0}".format(last_rank))
        return join_submatrices(filtered_data, filtered_data_compliment)

    LOG("Last Rank: {0}".format(last_rank))
    LOG("data size: {0}".format(len(data)))
    LOG("Filtered_data size: {0}".format(len(filtered_data)))

    sorted_compliment = ctwc_iteration(filtered_data_compliment, filtered_dist_matrix_compliment, cols_dist)

    sub_a = ctwc_iteration(np.squeeze(filtered_data.transpose()), cols_dist, filtered_dist_matrix) # Switching the order of cols/rows
    sub_b = ctwc_iteration(np.squeeze(sorted_compliment.transpose()), cols_dist, filtered_dist_matrix_compliment) # Switching the order of cols/rows

    return join_submatrices(np.squeeze(sub_a.transpose()), np.squeeze(sub_b.transpose()))


def ctwc(data, rows_dist, cols_dist):
    INFO("Running on data: {0}".format(data.shape))
    sorted_data = ctwc_iteration(data, rows_dist, cols_dist)
    INFO("Sorted data: {0}".format(sorted_data.shape))
    return sorted_data


def test():
    data, otus, samples = create_distance_matrix.get_sample_biom_table()
    tree = create_distance_matrix.get_gg_97_otu_tree()
    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, samples, tree, otus)

    sorted_data = ctwc(data, rows_dist, cols_dist)

if __name__ == "__main__":
    test()
