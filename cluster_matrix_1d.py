#!/usr/bin/python
from common import DEBUG,INFO,WARN,ERROR,FATAL
import warnings

import numpy as np
from sklearn.cluster import DBSCAN

import create_distance_matrix

sample_dist_matrix = np.array([    [ 0.0, 0.9, 0.1, 0.9, 0.1 ],
                                [ 0.9, 0.0, 0.9, 0.1, 0.9 ],
                                [ 0.1, 0.9, 0.0, 0.9, 0.1 ],
                                [ 0.1, 0.9, 0.9, 0.0, 0.9 ],
                                [ 0.1, 0.9, 0.1, 0.9, 0.0 ]
                                ])


def cluster_rows(data, dist_matrix, eps=0.5):
    a = data.tolist()
    db = DBSCAN(eps=eps, metric="precomputed", min_samples=3).fit(dist_matrix)
    for i in range(len(a)): a[i].insert(0, db.labels_[i])
    a.sort()
    for i in range(len(a)): del a[i][0]
    return np.array(a), db.labels_

def plot_results(vec):
    import matplotlib.pyplot as plt
    plt.plot(vec)
    plt.ylabel('some numbers')
    plt.show()

def plot_num_clusters_by_eps():
    best_i = 1
    largest_set = 0
    vec = []
    for i in range(2,50000, 150):
        clust, labels = cluster_rows(rows_dist, eps=1.0/i)
        cur_size = len(set(labels))
        if cur_size > largest_set:
            best_i = i
            largest_set = cur_size
        print "eps = {0} largest_set = {1}".format(1.0/i, cur_size)
        vec.append([1.0/i, cur_size])
    plot_results(vec)
def plot_distnace_matrix(dist_mat):
    mat = dist_mat.tolist()
    import matplotlib.pyplot as plt
    plt.matshow(mat, cmap=plt.cm.Blues)
    plt.show()

if __name__ == '__main__':
    data, otus, samples = create_distance_matrix.get_sample_biom_table()
    tree = create_distance_matrix.get_gg_97_otu_tree()
    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, samples, tree, otus)
    print "Original rows distance matrix:\n{0}".format(rows_dist)
    clust, labels = cluster_rows(data, rows_dist, eps=0.001)
    print "Clustered by rows ({0} clusters):\n{1}".format(len(set(labels)), clust)
    plot_distnace_matrix(data)
    plot_distnace_matrix(clust)
#    print cols_dist[0]
#    clust, labels = cluster_rows(cols_dist[0].transpose())
#    print clust.transpose()
#    print set(labels)
