#!/usr/bin/python

import numpy as np
from common import DEBUG,INFO,WARN,ERROR,FATAL,ASSERT,BP
import warnings
import sys

import math
import create_distance_matrix
import cluster_matrix_1d
import test_data

RECURSION_LIMIT = 100000


def get_left_child(children, node, n_leaves):
    return children[int(node - n_leaves)][0]

def get_right_child(children, node, n_leaves):
    return children[int(node - n_leaves)][1]

def get_node_depth(node, children, n_leaves, in_recursive=False):
    if node < n_leaves:
        return 0.0
    else:
        if not in_recursive:
            old_recursion_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(RECURSION_LIMIT)
        left = get_node_depth(get_left_child(children, node, n_leaves), children, n_leaves, True)
        right = get_node_depth(get_right_child(children, node, n_leaves), children, n_leaves, True)
        ret = 1 + left if left > right else 1 + right
        if not in_recursive:
            sys.setrecursionlimit(old_recursion_limit)
        return ret

def get_node_children_count(node, children, n_leaves, in_recursive=False):
    if node < n_leaves:
        return 0.0
    if not in_recursive:
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(RECURSION_LIMIT)
    count = 0.0;
    if get_left_child(children, node, n_leaves) != None:
        count += 1 + get_node_children_count(get_left_child(children, node, n_leaves), children, n_leaves, True)
    if get_right_child(children, node, n_leaves) != None:
        count += 1 + get_node_children_count(get_right_child(children, node, n_leaves), children, n_leaves, True)
    if not in_recursive:
        sys.setrecursionlimit(old_recursion_limit)
    return count

def get_all_labels_for_node(node, children, n_leaves, labels, in_recursive=False):
    if node < n_leaves:
        l = set([labels[node]])
    else:
        if not in_recursive:
            old_recursion_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(RECURSION_LIMIT)
        right_children = get_all_labels_for_node(get_right_child(children, node, n_leaves),
                                                 children,
                                                 n_leaves,
                                                 labels,
                                                 True)
        left_children = get_all_labels_for_node(get_left_child(children, node, n_leaves),
                                                children,
                                                n_leaves,
                                                labels,
                                                True)
        if not in_recursive:
            sys.setrecursionlimit(old_recursion_limit)
        l = right_children | left_children
    return l

def get_node_rank(node, children, n_leaves):
    depth = get_node_depth(node, children, n_leaves)
    count = get_node_children_count(node, children, n_leaves)
    if count == 0.0 or count == 1.0:
        return 0.0 + count / 2
    rank = depth / math.log(count, 2)
    return rank

def get_ranks(ag):
    n_samples = ag.children_.shape[0] + ag.n_leaves_
    l = []
    for i in range(n_samples):
        l.append([i,
                  get_node_depth(i, ag.children_, ag.n_leaves_),
                  get_node_children_count(i, ag.children_, ag.n_leaves_),
                  get_node_rank(i, ag.children_, ag.n_leaves_)])
    return l

def generate_dummy_data():
    from sklearn.cluster import AgglomerativeClustering
    import itertools
    X = np.array([[
         -5.27453240e-01,  -6.14130238e-01,  -1.63611427e+00,
         -9.26556498e-01,   7.82296885e-01,  -1.06286220e+00,
         -1.24368729e+00,  -1.16151964e+00,  -2.25816923e-01,
         -3.32354552e-02],
       [ -2.01273137e-01,   5.25758359e-01,   1.37940072e+00,
         -7.63256657e-01,  -1.27275323e+00,  -1.31618084e+00,
         -7.00167331e-01,   2.21410669e+00,   9.15456567e-01,
          7.93076923e-01],
       [  1.53249104e-01,  -5.48642411e-01,  -1.06559060e+00,
         -3.05253203e-01,  -1.93393495e+00,   1.39827978e-01,
          1.73359830e-01,   2.85576854e-02,  -1.19427027e+00,
          1.04395610e+00],
       [  1.00595172e+02,   1.01661346e+02,   1.00115635e+02,
          9.86884249e+01,   9.86506406e+01,   1.02214982e+02,
          1.01144087e+02,   1.00642778e+02,   1.01635339e+02,
          9.88981171e+01],
       [  1.01506262e+02,   1.00525318e+02,   9.93021764e+01,
          9.92514163e+01,   1.01199015e+02,   1.01771241e+02,
          1.00464097e+02,   9.97482396e+01,   9.96888274e+01,
          9.88297336e+01]])
    model = AgglomerativeClustering(linkage="average", affinity="cosine")
    model.fit(X)
    ii = itertools.count(X.shape[0])
    DEBUG(str([{'node_id': next(ii), 'left': x[0], 'right':x[1]} for x in model.children_]))
    return model, model.labels_

def get_nth_top_cluster_base_node(ranks_list, n=1):
    ranks_array = np.array(ranks_list)
    nth_max_index = -1
    for i in range(n):
        if nth_max_index >= 0:
            ranks_array = np.delete(ranks_array, nth_max_index, axis=0)
        max_indices = ranks_array.argmax(axis=0)
        nth_max_index = max_indices[3]
        nth_max_rank = ranks_array[nth_max_index][3]
    nth_max_node = ranks_array[nth_max_index][0]
    return nth_max_node, nth_max_rank, ranks_array[nth_max_index]

def get_index_list_for_label_filter(label_filter, labels):
    l = []
    for ind, label in enumerate(labels):
        if label in label_filter:
            l.append(ind)
    return l

def fix(array_like):
    return np.squeeze(np.asarray(array_like))

def filter_rows_by_top_rank(data, rows_dist, entry_names=None, debug=False):
    clust, labels, ag = cluster_matrix_1d.cluster_rows(data, rows_dist)
    BP()
    return _filter_rows_by_top_rank(data, rows_dist, clust, labels, ag, entry_names, debug)

def _filter_rows_by_top_rank(data, rows_dist, clust, labels, ag, entry_names=None, debug=False):
    log_func = INFO if debug else DEBUG
    ranks_list = get_ranks(ag)
    log_func("Lables: {0}".format(labels))
    log_func("Ranks: {0}".format(ranks_list))
    max_rank = get_nth_top_cluster_base_node(ranks_list)
    log_func("Max node: {0} rank: {1}".format(max_rank[0], max_rank[1]))
    labels_in_top_cluster = get_all_labels_for_node(max_rank[0],
                                                    ag.children_,
                                                    ag.n_leaves_,
                                                    labels)
    picked_indices = get_index_list_for_label_filter(labels_in_top_cluster, labels)
    log_func("Picked indices: {0}".format(picked_indices))

    filtered_data = [row for ind, row in enumerate(data) if ind in picked_indices]

    if entry_names is not None:
        entries = [ entry for ind, entry in enumerate(entry_names) if ind in picked_indices ]
        log_func("Picked entries: {0}".format(entries))

    filtered_data_compliment = [ row for ind, row in enumerate(data) if ind not in picked_indices]

    filtered_dist_matrix = [row for ind, row in enumerate(rows_dist) if ind in picked_indices]

    filtered_dist_matrix_compliment = [row for ind, row in enumerate(rows_dist) if ind not in picked_indices]

    return picked_indices, max_rank[1], fix(filtered_data), fix(filtered_dist_matrix), fix(filtered_data_compliment), fix(filtered_dist_matrix_compliment)

def filter_cols_by_top_rank(data, cols_dist, samples=None, debug=False):
    data = data.transpose()
    return filter_rows_by_top_rank(data, cols_dist, samples, debug)

def test():
    data, otus, samples = test_data.get_sample_biom_table()
    tree = test_data.get_gg_97_otu_tree()
    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, tree, samples, otus)

    picked_indices, max_rank, filtered_data, filtered_dist_matrix, _ , _ = filter_rows_by_top_rank(data, rows_dist, otus, True)

    ASSERT(len(picked_indices) == len(filtered_data))

    ASSERT(len(picked_indices) == len(filtered_dist_matrix))

    for row in filtered_data:
        ASSERT(row in data)

    clust, labels, ag = cluster_matrix_1d.cluster_rows(filtered_data.transpose(), cols_dist)

if __name__ == "__main__":
    test()
