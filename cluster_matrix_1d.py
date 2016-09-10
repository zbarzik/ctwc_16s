#!/usr/bin/python
from common import DEBUG,INFO,WARN,ERROR,FATAL,ASSERT,BP
import warnings

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import create_distance_matrix, test_data


# Constants
N_CLUSTERS = 16
SPC_BINARY_PATH = './SPC/'
SPC_BINARY_EXE = './SW'
SPC_TMP_FILES_PREFIX = '__tmp_ctwc'

# Simulates a distance matrix with two natural clusters. Expected result is (1,0,1,0,1).
sample_dist_matrix = np.array([ [ 0.0, 0.9, 0.1, 0.9, 0.1 ],
                                [ 0.9, 0.0, 0.9, 0.1, 0.9 ],
                                [ 0.1, 0.9, 0.0, 0.9, 0.1 ],
                                [ 0.1, 0.9, 0.9, 0.0, 0.9 ],
                                [ 0.1, 0.9, 0.1, 0.9, 0.0 ]
                                ])

def __spc__prepare_run_file(n_data_points):
    run_file = """
NumberOfPoints: {0}
DataFile: {1}.dat
Dimentions: 0
MinTemp:         0.10
MaxTemp:         1.0
TempStep:        0.01
OutFile:          {1}.out
SWCycles:        2000
KNearestNeighbours:    11
MSTree|
DirectedGrowth|
SaveSuscept|
WriteLables|
WriteCorFile~
DataIsMatrix: 1""".format(n_data_points, SPC_TMP_FILES_PREFIX)
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".run", "w+") as run_fn:
        run_fn.write(run_file)

def __spc_prepare_dat_file(dist_matrix):
    n_data_points = create_distance_matrix.generate_spc_output_files(dist_matrix, SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".dat")
    return n_data_points

def __spc_run_and_wait_for_completion():
    from subprocess import call
    args = [SPC_BINARY_EXE, SPC_TMP_FILES_PREFIX + ".run"]
    call(args, cwd=SPC_BINARY_PATH)

def __spc_parse_temperature_results():
    import collections

    NUM_CLUSTERS_IND = 3
    TEMP_IND = 1
    lines = []
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".out.dg_01", "r") as out_dg_fn:
        lines = out_dg_fn.readlines()
    DEBUG("SPC output:")
    for line in lines:
        DEBUG(line)
    num_clusters = []
    for line in lines:
        num_clusters.append(int(line.split()[NUM_CLUSTERS_IND]))
    counter=collections.Counter(num_clusters)
    candidates = counter.most_common(3)
    # most common might be completely frozen or completely dissolved.
    # 3 cases should include one sequence from the middle as well.
    candidate = candidates[0]
    for candidate in candidates:
        n_clusters, score = candidate
        if (n_clusters == int(lines[0].split()[NUM_CLUSTERS_IND]) or
            n_clusters == int(lines[-1].split()[NUM_CLUSTERS_IND])):
            continue # assuming that temperature range exceeds dynamic range - it's either frozen or dissolved
    for line in lines:
        if int(line.split()[NUM_CLUSTERS_IND]) == n_clusters:
            break
    temperature = float(line.split()[TEMP_IND])

    INFO("Most stable temperature: {0}. Number of clusters: {1}. Score: {2}".format(temperature, n_clusters, score))
    return temperature

def __spc_get_clusters_by_temperature(t):
    TEMP_IND = 1
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".out.dg_01.lab", "r") as out_dg_lab_fn:
        lines = out_dg_lab_fn.readlines()
    for line in lines:
        if float(line.split()[TEMP_IND]) == t:
            break
    return map(int, line.split()[TEMP_IND + 1:])

def __spc_get_cluster_members_by_cluster_id(clusters, cluster_id):
    clust = []
    for i, val in enumerate(clusters):
        if val == cluster_id:
            clust.append(i)
    return clust

def cluster_rows_spc(data, dist_matrix):
    DEBUG("Starting Super-Paramagnetic clustering...")
    n_data_points = __spc_prepare_dat_file(dist_matrix)
    __spc__prepare_run_file(n_data_points)
    __spc_run_and_wait_for_completion()
    t = __spc_parse_temperature_results()
    clusters = __spc_get_clusters_by_temperature(t)
    top_cluster = __spc_get_cluster_members_by_cluster_id(clusters, 0)
    DEBUG("Top cluster: {0}".format(top_cluster))
    DEBUG("Finished Super-Paramagnetic clustering.")
    return data, top_cluster, None


def cluster_rows_agglomerative(data, dist_matrix, n_clusters=N_CLUSTERS):
    DEBUG("Starting Agglomerative clustering...")
    ag = AgglomerativeClustering(n_clusters=n_clusters if n_clusters < len(dist_matrix) else len(dist_matrix),
                                 linkage="complete",
                                 affinity="precomputed").fit(dist_matrix)
    DEBUG("Finished Agglomerative clustering.")
    return data, ag.labels_, ag

def cluster_rows_dbscan(data, dist_matrix, eps=0.5):
    db = DBSCAN(eps=eps, metric="precomputed", min_samples=3).fit(dist_matrix)
    return data, db.labels_, db

def cluster_rows(data, dist_matrix):
    #return cluster_rows_dbscan(data, dist_matrix)
    #return cluster_rows_agglomerative(data, dist_matrix)
    return cluster_rows_spc(data, dist_matrix)

def plot_results(vec):
    import matplotlib.pyplot as plt
    plt.plot(vec)
    plt.show()

def plot_num_clusters_by_eps(data, cols_dist, rows_dist):
    best_i = 1
    largest_set = 0
    vec = []
    for i in range(2, 20000, 100):
        clust, labels = cluster_rows_dbscan(data.transpose(), cols_dist, eps=1.0/i)
        cur_size = len(set(labels))
        if cur_size > largest_set:
            best_i = i
            largest_set = cur_size
        DEBUG("eps = {0} largest_set = {1}".format(1.0/i, cur_size))
        vec.append([1.0/i, cur_size])
    plot_results(vec)

def plot_distnace_matrix(dist_mat):
    mat = dist_mat.tolist()
    import matplotlib.pyplot as plt
    plt.matshow(mat, cmap=plt.cm.Blues)
    plt.show()

def test_agglomerative_clustering():
    _, labels, ag = cluster_rows_agglomerative(None, sample_dist_matrix, 2)
    import rank_cluster
    ranks_list = rank_cluster.get_ranks(ag)
    rank_cluster.get_nth_top_cluster_base_node(ranks_list)

def test_dbscan_clustering(data, dist_matrix):
    _, labels, ag = cluster_rows_dbscan(None, dist_matrix)

def test():
    #test_agglomerative_clustering()

    data, otus, samples = test_data.get_sample_biom_table()

    #data = inject_row_pattern_to_data(data)

    #data = inject_col_pattern_to_data(data)

    INFO("Original data:\n{0}\n\n".format(data))

    tree = test_data.get_gg_97_otu_tree()

    rows_dist, cols_dist = create_distance_matrix.get_distance_matrices(data, tree, samples, otus)

    ASSERT(rows_dist.shape[0] == rows_dist.shape[1])
    ASSERT(cols_dist.shape[0] == cols_dist.shape[1])

    ASSERT(rows_dist.shape[0] == data.shape[0])
    ASSERT(cols_dist.shape[0] == data.shape[1])

    INFO("Original cols distance matrix:\n{0}\n\n".format(cols_dist))

    cluster_rows_spc(data, cols_dist)


    #test_dbscan_clustering(data, cols_dist)

    clust, labels, _ = cluster_rows(data.transpose(), cols_dist)

    st = "Labels:\n"
    for i in range(labels.shape[0]):
        st += str(labels[i]) + " "
    INFO(st + "\n\n")

    INFO("Clustered by cols ({0} clusters):\n{1}\n\n".format(len(set(labels)), clust))
    #plot_distnace_matrix(data)
    #plot_distnace_matrix(clust)

    INFO("Original rows distance matrix:\n{0}\n\n".format(rows_dist))

    clust, labels, _ = cluster_rows(data, rows_dist)

    st = "Labels:\n"
    for i in range(labels.shape[0]):
        st += str(labels[i]) + " "
    INFO(st + "\n\n")

    INFO("Clustered by rows ({0} clusters):\n{1}\n\n".format(len(set(labels)), clust))
    #plot_distnace_matrix(data)
    #plot_distnace_matrix(clust.transpose())

def inject_row_pattern_to_data(data):
    z = np.zeros(data.shape)
    for col in range(data.shape[1]): # Flooding to create a false relationship between rows
        for row in range(6):
            z[row, col] = (col % 2) * 200
        for row in range(data.shape[0] - 10, data.shape[0]):
            z[row, col] += (1 - (col % 2)) * 200
    return data + z

def inject_col_pattern_to_data(data):
    z = np.zeros(data.shape)
    for col in range(30):
        for row in range(data.shape[0]): # Flooding to create a false relationship between rows
            z[row, col] = data[row, 0] - data[row, col]
    return data + z

if __name__ == "__main__":
    test()
