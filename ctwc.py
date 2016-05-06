#!/usr/bin/python
from common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import create_distance_matrix, rank_cluster
import numpy as np



def join_submatrices(mat_a, mat_b):
    if len(mat_b) == 0:
        return mat_a
    if len(mat_a) == 0:
        return mat_b
    axis = 1 if mat_a.shape[0] == mat_b.shape[0] else 0
    ret = np.concatenate((mat_a, mat_b), axis)
    ASSERT(ret.shape[axis] == mat_a.shape[axis] + mat_b.shape[axis])
    return np.squeeze(ret)

def ctwc_bicluster_iteration(data, rows_dist, cols_dist):
    DEBUG("Now running on data size: {0}".format(data.shape))
    picked_indices, last_rank, filtered_data, filtered_dist_matrix, filtered_data_compliment, filtered_dist_matrix_compliment = rank_cluster.filter_rows_by_top_rank(
                                                                                                                                                     data, rows_dist)
    if len(filtered_data_compliment) == 0:
        DEBUG("Didn't filter out anything. last_rank = {0}".format(last_rank))
        return join_submatrices(filtered_data, filtered_data_compliment)

    DEBUG("Last Rank: {0}".format(last_rank))
    DEBUG("data size: {0}".format(len(data)))
    DEBUG("Filtered_data size: {0}".format(len(filtered_data)))

    sorted_compliment = ctwc_bicluster_iteration(filtered_data_compliment, filtered_dist_matrix_compliment, cols_dist)

    sub_a = ctwc_bicluster_iteration(np.squeeze(filtered_data.transpose()), cols_dist, filtered_dist_matrix) # Switching the order of cols/rows
    sub_b = ctwc_bicluster_iteration(np.squeeze(sorted_compliment.transpose()), cols_dist, filtered_dist_matrix_compliment) # Switching the order of cols/rows

    return join_submatrices(np.squeeze(sub_a.transpose()), np.squeeze(sub_b.transpose()))


def ctwc_bicluster(data, rows_dist, cols_dist):
    DEBUG("Running on data: {0}".format(data.shape))
    sorted_data = ctwc_bicluster_iteration(data, rows_dist, cols_dist)
    DEBUG("Sorted data: {0}".format(sorted_data.shape))
    return sorted_data

def prepare_otu_filters_from_indices(picked_indices, otus):
    selected_rows_filter = [ otu for index, otu in enumerate(otus) if index not in picked_indices ]
    compliment_rows_filter = [ otu for index, otu in enumerate(otus) if index in picked_indices ]
    return selected_rows_filter, compliment_rows_filter

def prepare_sample_filters_from_indices(picked_indices, samples):
    selected_cols_filter = [ samp for index, samp in enumerate(samples) if index not in picked_indices ]
    compliment_cols_filter = [ samp for index, samp in enumerate(samples) if index in picked_indices ]
    return selected_cols_filter, compliment_cols_filter

def ctwc_select(data, tree, samples, otus):
    INFO("Preparing distance matrices for full data...")
    rows_dist_1, _ = create_distance_matrix.get_distance_matrices(data, tree, samples, otus)
    INFO("Iteration 1: Picking rows...")
    picked_indices_1, last_rank_1, _, _, _, _ = rank_cluster.filter_rows_by_top_rank(data, rows_dist_1)
    selected_rows_filter_1, compliment_rows_filter_1 = prepare_otu_filters_from_indices(picked_indices_1, otus)

    INFO("Iteration 1.1: Picking columns from selected rows...")
    _, cols_dist_1_1 = create_distance_matrix.get_distance_matrices(data, tree, samples, otus, otu_filter=selected_rows_filter_1)
    picked_indices_1_1, last_rank_1_1, _, _, _, _ = rank_cluster.filter_cols_by_top_rank(data, cols_dist_1_1)

    INFO("Iteration 1.2: Picking columns from rows compliment...")
    _, cols_dist_1_2 = create_distance_matrix.get_distance_matrices(data, tree, samples, otus, otu_filter=compliment_rows_filter_1)
    picked_indices_1_2, last_rank_1_2, _, _, _, _ = rank_cluster.filter_cols_by_top_rank(data, cols_dist_1_2)
    selected_cols_filter_1_2, compliment_cols_filter_1_2 = prepare_sample_filters_from_indices(picked_indices_1_2, samples)

    INFO("Iteration 2: Picking rows based on selected columns...")
    selected_cols_filter_1_1, compliment_cols_filter_1_1 = prepare_sample_filters_from_indices(picked_indices_1_1, samples)
    rows_dist_2, _ = create_distance_matrix.get_distance_matrices(data, tree, samples, otus, sample_filter=selected_cols_filter_1_1)
    picked_indices_2, last_rank_2, _, _, _, _ = rank_cluster.filter_rows_by_top_rank(data, rows_dist_2)
    selected_rows_filter_2, compliment_rows_filter_2 = prepare_otu_filters_from_indices(picked_indices_2, otus)

    output = { "Iteration 1.1" : (compliment_rows_filter_1, compliment_cols_filter_1_1),
               "Iteration 1.2" : (selected_rows_filter_1, compliment_cols_filter_1_2),
               "Iteration 2"   : (compliment_rows_filter_2, compliment_cols_filter_1_1) }
    return output

def test():
    np.seterr(all="ignore")
    data, otus, samples = create_distance_matrix.get_sample_biom_table()
    tree = create_distance_matrix.get_gg_97_otu_tree()

    #rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, tree, samples, otus)
    #sorted_data = ctwc_bicluster(data, rows_dist, cols_dist)

    output = ctwc_select(data, tree, samples, otus)
    INFO("Full data size: {0} X {1}".format(data.shape[0], data.shape[1]))
    for elem in output:
        INFO("{0}: {1} X {2}".format(elem, len(output[elem][0]), len(output[elem][1])))

if __name__ == "__main__":
    test()
