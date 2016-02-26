#!/usr/bin/python

from sklearn.cluster import ward_tree
import numpy as np
import itertools
from common import DEBUG,INFO,WARN,ERROR,FATAL,ASSERT,BP
import warnings

import math
import create_distance_matrix
import cluster_matrix_1d

def get_node_depth(node, children, n_leaves):
    if node < n_leaves:
        return 0
    else:
        left = get_node_depth(children[node - n_leaves][0], children, n_leaves)
        right = get_node_depth(children[node - n_leaves][1], children, n_leaves)
        return 1 + left if left > right else 1 + right

def get_node_children_count(node, children, n_leaves):
    if node < n_leaves:
        return 0
    else:
        left_count = get_node_children_count(children[node - n_leaves][0], children, n_leaves)
        right_count = get_node_children_count(children[node - n_leaves][1], children, n_leaves)
        return 1 + left_count + right_count

def get_node_rank(node, children, n_leaves):
    depth = get_node_depth(node, children, n_leaves)
    count = get_node_children_count(node, children, n_leaves)
    if count == 0:
        return 0
    log_count = math.log(count, 2)
    if log_count == 0:
        return 0
    rank = depth / math.log(count, 2)
    return rank

def get_ranks(ag):
    n_samples = ag.children_.shape[0] + ag.n_leaves_
    l = []
    for i in range(n_samples):
        l.append(get_node_rank(i, ag.children_, ag.n_leaves_))
    return l

def test():
    data, otus, samples = create_distance_matrix.get_sample_biom_table()
    tree = create_distance_matrix.get_gg_97_otu_tree()
    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, samples, tree, otus)
    clust, labels, ag = cluster_matrix_1d.cluster_rows(data, rows_dist)
    print get_ranks(ag)
    clust, labels, ag = cluster_matrix_1d.cluster_rows(data.transpose(), cols_dist)
  #  print get_sample_tree()

if __name__ == "__main__":
    test()
