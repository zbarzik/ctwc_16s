#!/usr/bin/python
from common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import create_distance_matrix, rank_cluster

def ctwc(data, rows_dist, cols_dist):
    last_rank = 10000
    rows = True

    while True:
        INFO("Now running on data size: {0}".format(len(data)))
        filtered_data, filtered_dist_matrix, picked_indices, last_rank = rank_cluster.filter_rows_by_top_rank(data, rows_dist if rows else cols_dist)
        if last_rank < 1.2:
            break
        if len(filtered_data) == len(data):
            INFO("Didn't filter out anything")
            break
        INFO("Last Rank: {0}".format(last_rank))
        INFO("data size: {0}".format(len(data)))
        INFO("Filtered_data size: {0}".format(len(filtered_data)))
        data = filtered_data.transpose()
        if rows:
            rows_dist = filtered_dist_matrix
        else:
            cols_dist = filtered_dist_matrix
        rows = not rows


def test():
    data, otus, samples = create_distance_matrix.get_sample_biom_table()
    tree = create_distance_matrix.get_gg_97_otu_tree()
    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, samples, tree, otus)

    ctwc(data, rows_dist, cols_dist)


if __name__ == "__main__":
    test()
