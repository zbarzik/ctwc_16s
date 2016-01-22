#!/usr/bin/python
from common import DEBUG,INFO,WARN,ERROR,FATAL
import warnings

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import create_distance_matrix

# Simulates a distance matrix with two natural clusters. Expected result is (1,0,1,0,1). 
sample_dist_matrix = np.array([ [ 0.0, 0.9, 0.1, 0.9, 0.1 ],
                                [ 0.9, 0.0, 0.9, 0.1, 0.9 ],
                                [ 0.1, 0.9, 0.0, 0.9, 0.1 ],
                                [ 0.1, 0.9, 0.9, 0.0, 0.9 ],
                                [ 0.1, 0.9, 0.1, 0.9, 0.0 ]
                                ])


def cluster_rows_agglomerative(data, dist_matrix, n_clusters=10):
    a = data.tolist()
    ag = AgglomerativeClustering(n_clusters=n_clusters).fit(dist_matrix)
    for i in range(len(a)): a[i].insert(0, ag.labels_[i])
    a.sort()
    for i in range(len(a)): del a[i][0]
    return np.array(a), ag.labels_

def cluster_rows_dbscan(data, dist_matrix, eps=0.5):
    a = data.tolist()
    db = DBSCAN(eps=eps, metric="precomputed", min_samples=3).fit(dist_matrix)
    for i in range(len(a)): a[i].insert(0, db.labels_[i])
    a.sort()
    for i in range(len(a)): del a[i][0]
    return np.array(a), db.labels_

def cluster_rows(data, dist_matrix):
    return cluster_rows_agglomerative(data, dist_matrix)

def plot_results(vec):
    import matplotlib.pyplot as plt
    plt.plot(vec)
    plt.show()

def plot_num_clusters_by_eps(data, cols_dist, rows_dist):
    best_i = 1
    largest_set = 0
    vec = []
    for i in range(2, 20000, 100):
        clust, labels = cluster_rows_dbscan(data.transpose(), cols_dist[0], eps=1.0/i)
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

def test():
    data, otus, samples = create_distance_matrix.get_sample_biom_table()

    data = inject_pattern_to_data(data)

    tree = create_distance_matrix.get_gg_97_otu_tree()
    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, samples, tree, otus)

    print "Original rows distance matrix:\n{0}\n\n".format(rows_dist)

    clust, labels = cluster_rows(data, rows_dist)

    st = "Lables:\n"
    for i in range(labels.shape[0]):
        st += str(labels[i])
    print st + "\n\n"

    print "Clustered by rows ({0} clusters):\n{1}\n\n".format(len(set(labels)), clust)
    #plot_distnace_matrix(data)
    #plot_distnace_matrix(clust)

    print "Original cols distance matrix:\n{0}\n\n".format(cols_dist)

    clust, labels = cluster_rows(data.transpose(), cols_dist[0])

    st = "Lables:\n"
    for i in range(labels.shape[0]):
        st += str(labels[i])
    print st + "\n\n"

    print "Clustered by rows ({0} clusters):\n{1}\n\n".format(len(set(labels)), clust.transpose())
    #plot_distnace_matrix(data)
    #plot_distnace_matrix(clust.transpose())

def inject_pattern_to_data(data):
    data, otus, samples = create_distance_matrix.get_sample_biom_table()
    z = np.zeros(data.shape)
    for col in range(len(samples)): # Add a lot of some bacteria to some of the samples
        row = 0
        z[row][col] += (col % 2) * 150
        z[row+1][col] += (col % 2) * 200
        z[row+2][col] += (1 - (col % 2)) * 200
        z[row+3][col] += (1 - (col % 2)) * 200
    return data + z

if __name__ == "__main__":
    test()
