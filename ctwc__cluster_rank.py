#!/usr/bin/python

import numpy as np
from ctwc__common import DEBUG,INFO,WARN,ERROR,FATAL,ASSERT,BP
import warnings
import sys

import math
import ctwc__distnace_matrix
import ctwc__cluster_1d
import ctwc__data_handler

RECURSION_LIMIT = 100000

def __get_parent(children, node, n_leaves):
    for i in range(n_leaves, len(children) + n_leaves):
        if (__get_left_child(children, i, n_leaves) == node or
            __get_right_child(children, i, n_leaves) == node):
            return i
    return None

def __get_left_child(children, node, n_leaves):
    return children[int(node - n_leaves)][0]

def __get_right_child(children, node, n_leaves):
    return children[int(node - n_leaves)][1]

def __get_node_depth(node, children, n_leaves, in_recursive=False):
    if node < n_leaves:
        return 0.0
    else:
        if not in_recursive:
            old_recursion_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(RECURSION_LIMIT)
        left = __get_node_depth(__get_left_child(children, node, n_leaves), children, n_leaves, True)
        right = __get_node_depth(__get_right_child(children, node, n_leaves), children, n_leaves, True)
        ret = 1 + left if left > right else 1 + right
        if not in_recursive:
            sys.setrecursionlimit(old_recursion_limit)
        return ret

def __get_node_children_count(node, children, n_leaves, in_recursive=False):
    if node < n_leaves:
        return 0.0
    if not in_recursive:
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(RECURSION_LIMIT)
    count = 0.0;
    if __get_left_child(children, node, n_leaves) != None:
        count += 1 + __get_node_children_count(__get_left_child(children, node, n_leaves), children, n_leaves, True)
    if __get_right_child(children, node, n_leaves) != None:
        count += 1 + __get_node_children_count(__get_right_child(children, node, n_leaves), children, n_leaves, True)
    if not in_recursive:
        sys.setrecursionlimit(old_recursion_limit)
    return count

def __get_all_labels_for_node(node, children, n_leaves, labels, in_recursive=False):
    if node < n_leaves:
        l = set([labels[node]])
    else:
        if not in_recursive:
            old_recursion_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(RECURSION_LIMIT)
        right_children = __get_all_labels_for_node(__get_right_child(children, node, n_leaves),
                                                 children,
                                                 n_leaves,
                                                 labels,
                                                 True)
        left_children = __get_all_labels_for_node(__get_left_child(children, node, n_leaves),
                                                children,
                                                n_leaves,
                                                labels,
                                                True)
        if not in_recursive:
            sys.setrecursionlimit(old_recursion_limit)
        l = right_children | left_children
    return l

def __verify_node(node, children, n_leaves):
    parent = __get_parent(children, node, n_leaves)
    root = len(children) + n_leaves - 1
    ASSERT(parent is None and node == root or
           parent is not None)
    ASSERT(parent is None or
           __get_left_child(children, parent, n_leaves) == node or
           __get_right_child(children, parent, n_leaves) == node)

def get_node_rank(node, children, n_leaves):
    #return __get_node_rank__depth_to_log_children(node, children, n_leaves)
    return __get_node_rank__size_of_sibling(node, children, n_leaves)

def __get_node_rank__size_of_sibling(node, children, n_leaves):
    count = __get_node_children_count(node, children, n_leaves)
    parent = __get_parent(children, node, n_leaves)
    if parent is not None:
        parent_count = __get_node_children_count(parent, children, n_leaves)
        return (count, (count / parent_count))
    else:
        return (count, 0.0)

def __get_node_rank__depth_to_log_children(node, children, n_leaves):
    depth = __get_node_depth(node, children, n_leaves)
    count = __get_node_children_count(node, children, n_leaves)
    if count == 0.0 or count == 1.0:
        return 0.0 + count / 2.0
    rank = depth / math.log(count, 2)
    return rank

def __get_ranks_agglomerative(ag):
    n_samples = ag.children_.shape[0] + ag.n_leaves_
    l = []
    for i in range(n_samples):
        l.append([i,
                  __get_node_depth(i, ag.children_, ag.n_leaves_),
                  __get_node_children_count(i, ag.children_, ag.n_leaves_),
                  get_node_rank(i, ag.children_, ag.n_leaves_)])
    return l

def __generate_dummy_data():
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

def __is_greater(val1, val2):
    if not hasattr(val1, '__iter__'):
        return val1 > val2
    for i in range(len(val1)):
        if val1[i] == val2[i]:
            continue
        else:
            return val1[i] > val2[i]
    return False

def __get_max_value_in_array(array):
    max_val = None
    max_ind = 0
    for i, val in enumerate(array):
        if max_val is None:
            max_val = val
            max_ind = i
        elif __is_greater(val, max_val):
            max_val = val
            max_ind = i
    return max_ind, max_val

def __get_nth_top_cluster_base_node(ranks_list, n=1):
    ranks_array = np.array(ranks_list)
    nth_max_index = -1
    for i in range(n):
        if nth_max_index >= 0:
            ranks_array = np.delete(ranks_array, nth_max_index, axis=0)
        nth_max_index, nth_max_rank = __get_max_value_in_array(ranks_array)
    nth_max_node = ranks_array[nth_max_index][0]
    return nth_max_node, nth_max_rank, ranks_array[nth_max_index]

def __get_index_list_for_label_filter(label_filter, labels):
    l = []
    for ind, label in enumerate(labels):
        if label in label_filter:
            l.append(ind)
    return l

def __fix(array_like):
    return np.squeeze(np.asarray(array_like))

def filter_rows_by_top_rank(data, rows_dist, entry_names=None, debug=False):
    DEBUG("Starting to cluster data...")
    clust, labels, ag = ctwc__cluster_1d.cluster_rows(data, rows_dist)
    INFO("Clustered labels: {0}".format(labels))
    return __filter_rows_by_top_rank(data, rows_dist, clust, labels, ag, entry_names, debug)

def __filter_rows_by_top_rank(data, rows_dist, clust, labels, ag, entry_names=None, debug=False):
    log_func = INFO if debug else DEBUG
    if ag is not None:
        ranks_list = __get_ranks_agglomerative(ag)
        log_func("Labels: {0}".format(labels))
        log_func("Ranks: {0}".format(ranks_list))
        max_rank = __get_nth_top_cluster_base_node(ranks_list)
        if max_rank[0] == len(ranks_list) - 1:
            max_rank = __get_nth_top_cluster_base_node(ranks_list, 2)
        log_func("Max node: {0} rank: {1}".format(max_rank[0], max_rank[1]))
        labels_in_top_cluster = __get_all_labels_for_node(max_rank[0],
                                                        ag.children_,
                                                        ag.n_leaves_,
                                                        labels)
        picked_indices = __get_index_list_for_label_filter(labels_in_top_cluster, labels)
    else:
        max_rank = (-1, -1)
        picked_indices = labels
    log_func("Picked indices: {0}".format(picked_indices))

    filtered_data = [row for ind, row in enumerate(data) if ind in picked_indices]

    if entry_names is not None:
        entries = [ entry for ind, entry in enumerate(entry_names) if ind in picked_indices ]
        log_func("Picked entries: {0}".format(entries))

    filtered_data_compliment = [ row for ind, row in enumerate(data) if ind not in picked_indices]

    filtered_dist_matrix = [row for ind, row in enumerate(rows_dist) if ind in picked_indices]

    filtered_dist_matrix_compliment = [row for ind, row in enumerate(rows_dist) if ind not in picked_indices]

    return picked_indices, max_rank[1], __fix(filtered_data), __fix(filtered_dist_matrix), __fix(filtered_data_compliment), __fix(filtered_dist_matrix_compliment)

def filter_cols_by_top_rank(data, cols_dist, samples=None, debug=False):
    data = data.transpose()
    return filter_rows_by_top_rank(data, cols_dist, samples, debug)

def test():
    data, otus, samples = ctwc__data_handler.get_sample_biom_table()
    tree = ctwc__data_handler.get_gg_97_otu_tree()
    _, cols_dist = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus, skip_rows=True)
    picked_indices, max_rank, filtered_data, filtered_dist_matrix, _ , _ = filter_cols_by_top_rank(data, cols_dist, otus, True)

    INFO("Picked {0} indices".format(len(picked_indices)))
    clust, labels, ag = ctwc__cluster_1d.cluster_rows(filtered_data.transpose(), cols_dist)

if __name__ == "__main__":
    test()
