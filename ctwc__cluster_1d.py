#!/usr/bin/python
from ctwc__common import DEBUG,INFO,WARN,ERROR,FATAL,ASSERT,BP,save_to_file,load_from_file
import warnings

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import ctwc__distnace_matrix, ctwc__data_handler


# Constants
N_CLUSTERS = 16
SPC_BINARY_PATH = './SPC/'
SPC_BINARY_EXE = './SW'
SPC_TMP_FILES_PREFIX = '__tmp_ctwc'
SPC_CLUSTER_FILE = './spc_cluster-{0}.pklz'


# Simulates a distance matrix with two natural clusters. Expected result is (1,0,1,0,1).
sample_dist_matrix = np.array([ [ 0.0, 0.9, 0.1, 0.9, 0.1 ],
                                [ 0.9, 0.0, 0.9, 0.1, 0.9 ],
                                [ 0.1, 0.9, 0.0, 0.9, 0.1 ],
                                [ 0.1, 0.9, 0.9, 0.0, 0.9 ],
                                [ 0.1, 0.9, 0.1, 0.9, 0.0 ]
                                ])


def __spc_prepare_run_file(n_data_points):
    run_file = """
NumberOfPoints: {0}
NumberOfEdges: {2}
DataFile: {1}.dat
MinTemp: 0.0
MaxTemp: 0.3
TempStep: 0.001
OutFile: {1}.out
SWCycles: 2000
KNearestNeighbours: 11
Dimension: 0
MSTree|
NearestNeighbours|
DirectedGrowth|
SaveSuscept|
WriteLables|
WriteCorFile|
""".format(n_data_points, SPC_TMP_FILES_PREFIX, n_data_points**2/2)
    INFO(run_file)
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".run", "w+") as run_f:
        run_f.write(run_file)

def __spc_prepare_dat_file(dist_mat, output_filename):
    with open(output_filename, 'w+') as fn:
        for r in range(dist_mat.shape[0]):
            for c in range(dist_mat.shape[1]):
                fn.write("{0} {1} {2}\n".format(r + 1, c + 1, dist_mat[r][c]))
    return dist_mat.shape[0]

def __spc_prepare_dat_file(dist_mat):
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".dat", 'w+') as fn:
        for r in range(dist_mat.shape[0]):
            for c in range(dist_mat.shape[1]):
                fn.write("{0} {1} {2}\n".format(r + 1, c + 1, dist_mat[r][c]))
    return dist_mat.shape[0]

def __spc_prepare_edge_file(n):
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".edge", "w+") as edge_f:
        for r in range(1, n):
            for c in range(1, r):
                edge_f.write("{0} {1}\n".format(r, c))

def __spc_run_and_wait_for_completion():
    from subprocess import call
    args = [SPC_BINARY_EXE, SPC_TMP_FILES_PREFIX + ".run"]
    call(args, cwd=SPC_BINARY_PATH)

def __pick_line_by_most_stable_largest_cluster(lines, lower_threshold=0, upper_threshold=float('Inf')):
    import collections
    LARGEST_CLUSTER_IND = 4
    largest_cluster_sizes = []
    for line in lines:
        largest_cluster_sizes.append(int(line.split()[LARGEST_CLUSTER_IND]))
    counter=collections.Counter(largest_cluster_sizes)
    candidates = counter.most_common(200)
    # most common might be completely frozen or completely dissolved.
    # 3 cases should include one sequence from the middle as well.
    found = False
    for candidate in candidates:
        largest_cluster, score = candidate
        if (largest_cluster == int(lines[0].split()[LARGEST_CLUSTER_IND]) or
            largest_cluster == int(lines[-1].split()[LARGEST_CLUSTER_IND]) or
            largest_cluster < lower_threshold or
            largest_cluster > upper_threshold or
            score < 3):
            continue # assuming that temperature range exceeds dynamic range - it's either frozen or dissolved
        found = True
        break

    if not found: # we didn't find anything that matches the thresholds, return something good enough
        candidates = counter.most_common(200)
        for candidate in candidates:
            largest_cluster, score = candidate
            if (largest_cluster == int(lines[0].split()[LARGEST_CLUSTER_IND]) or
                largest_cluster == int(lines[-1].split()[LARGEST_CLUSTER_IND])):
                continue
            break

    for line in lines:
        if int(line.split()[LARGEST_CLUSTER_IND]) == largest_cluster:
            break
    return line

def __pick_line_by_num_clusters(lines):
    import collections
    NUM_CLUSTERS_IND = 3
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
        break
    for line in lines:
        if int(line.split()[NUM_CLUSTERS_IND]) == n_clusters:
            break
    return line

def __spc_parse_temperature_results(data_points):
    TEMP_IND = 1
    lines = []
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".out.dg_01", "r") as out_dg_fn:
        lines = out_dg_fn.readlines()
    DEBUG("SPC output:")
    for line in lines:
        DEBUG(line)

    #line = __pick_line_by_num_clusters(lines)
    lower_threshold = max(50.0, data_points / 200.0) # 0.5% or 50
    upper_threshold = min(10000.0, data_points / 2.0) # 50% or 10000
    line = __pick_line_by_most_stable_largest_cluster(lines, lower_threshold, upper_threshold)
    temperature = float(line.split()[TEMP_IND])

    INFO("Most stable temperature: {0}".format(temperature))
    return temperature, lines

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

def __spc_clear_temporary_files():
    from glob import glob
    import os
    map(os.remove, glob(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + '*'))

def __get_precalculated_spc_file_if_exists(h):
    return load_from_file(SPC_CLUSTER_FILE.format(h))

def __calculate_hash_for_data(data, dist_matrix):
    return hash(str([ hash(data.tostring()), hash(str(dist_matrix)) ]) ) # eh close enough

def __get_precalculated_spc_file_if_exists_for_data(data, dist_matrix):
    h = __calculate_hash_for_data(data, dist_matrix)
    return __get_precalculated_spc_file_if_exists(h)

def __save_calculated_spc_file_and_hash_for_data(data, dist_matrix, cluster, log):
    DEBUG("Saving calculated SPC cluster to file...")
    h = __calculate_hash_for_data(data, dist_matrix)
    save_to_file((cluster, log), SPC_CLUSTER_FILE.format(h))

def cluster_rows_spc(data, dist_matrix):
    DEBUG("Checking for cached results...")
    cached_results = __get_precalculated_spc_file_if_exists_for_data(data, dist_matrix)
    if cached_results is not None:
        cached_cluster, log = cached_results
        INFO("Using pre-cached SPC results for input")
        for line in log:
            INFO(line)
        return data, cached_cluster, None

    DEBUG("Clearing temporary files from previous runs...")
    __spc_clear_temporary_files()
    DEBUG("Starting Super-Paramagnetic clustering...")
    n_data_points = __spc_prepare_dat_file(dist_matrix)
    __spc_prepare_edge_file(n_data_points)
    __spc_prepare_run_file(n_data_points)
    __spc_run_and_wait_for_completion()
    t, log = __spc_parse_temperature_results(n_data_points)
    clusters = __spc_get_clusters_by_temperature(t)
    top_cluster = __spc_get_cluster_members_by_cluster_id(clusters, 0)
    DEBUG("Top cluster: {0}".format(top_cluster))
    DEBUG("Saving results in cache...")
    __save_calculated_spc_file_and_hash_for_data(data, dist_matrix, top_cluster, log)
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
    return cluster_rows_spc(data, dist_matrix)

def test_agglomerative_clustering():
    _, labels, ag = cluster_rows_agglomerative(None, sample_dist_matrix, 2)
    import ctwc__cluster_rank
    ranks_list = ctwc__cluster_rank.get_ranks(ag)
    ctwc__cluster_rank.get_nth_top_cluster_base_node(ranks_list)

def test_dbscan_clustering(data, dist_matrix):
    _, labels, ag = cluster_rows_dbscan(None, dist_matrix)

def test_spc_clustering():
    SIZE = 100
    z = np.zeros((SIZE, SIZE))
    for i in range(0, SIZE):
        for j in range(0, SIZE):
            z[i][j] = 0.1 if (i+j) % 2 == 0 else 0.8
            if i == j:
                z[i][j] = 0.0
    INFO(z)
    _, top_cluster, _ = cluster_rows_spc(None, z)
    INFO(top_cluster)

def test():
    test_spc_clustering()
    return

    data, otus, samples = ctwc__data_handler.get_sample_biom_table()

    INFO("Original data:\n{0}\n\n".format(data))

    tree = ctwc__data_handler.get_gg_97_otu_tree()

    rows_dist, cols_dist = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus)

    ASSERT(rows_dist.shape[0] == rows_dist.shape[1])
    ASSERT(cols_dist.shape[0] == cols_dist.shape[1])

    ASSERT(rows_dist.shape[0] == data.shape[0])
    ASSERT(cols_dist.shape[0] == data.shape[1])

    INFO("Original cols distance matrix:\n{0}\n\n".format(cols_dist))

    cluster_rows_spc(data, cols_dist)

    clust, labels, _ = cluster_rows(data.transpose(), cols_dist)

    st = "Labels:\n"
    for i in range(labels.shape[0]):
        st += str(labels[i]) + " "
    INFO(st + "\n\n")

    INFO("Clustered by cols ({0} clusters):\n{1}\n\n".format(len(set(labels)), clust))

    INFO("Original rows distance matrix:\n{0}\n\n".format(rows_dist))

    clust, labels, _ = cluster_rows(data, rows_dist)

    st = "Labels:\n"
    for i in range(labels.shape[0]):
        st += str(labels[i]) + " "
    INFO(st + "\n\n")

    INFO("Clustered by rows ({0} clusters):\n{1}\n\n".format(len(set(labels)), clust))

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
