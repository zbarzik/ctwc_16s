#!/usr/bin/python
from ctwc__common import DEBUG,INFO,WARN,ERROR,FATAL,ASSERT,BP,save_to_file,load_from_file
import warnings

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import ctwc__distance_matrix, ctwc__data_handler


# Constants
N_CLUSTERS = 16
SPC_BINARY_PATH = './SPC/'
SPC_BINARY_EXE = './SW'
SPC_TMP_FILES_PREFIX = '__tmp_ctwc'
SPC_CLUSTER_FILE = './spc_cluster-{0}.pklz'
ALLOW_CACHING = False

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

def __spc_prepare_dat_file(dist_mat):
    lines = []
    for r in xrange(dist_mat.shape[0]):
        for c in xrange(dist_mat.shape[1]):
            lines.append("{0} {1} {2}\n".format(r + 1, c + 1, dist_mat[r][c]))
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".dat", 'w+') as fn:
        fn.write("".join(lines))
    return dist_mat.shape[0]

def __spc_get_non_masked_data_points(dist_mat):
    n = dist_mat.shape[0]
    tmp = np.zeros(dist_mat.shape)
    tmp[dist_mat < ctwc__distance_matrix.INF_VALUE] = 1
    np.fill_diagonal(tmp, 0.0)
    non_zero_rows = len(set(tmp.nonzero()[0]))
    INFO("Non-masked rows: {0}".format(non_zero_rows))
    return non_zero_rows

def __spc_prepare_edge_file(n):
    lines = []
    for r in xrange(1, n):
       for c in xrange(1, r):
           lines.append("{0} {1}\n".format(r, c))

    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".edge", "w+") as edge_f:
        edge_f.write("".join(lines))

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
                score < 3):
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

def __spc_parse_temperature_results(non_masked_data_points, cluster_limit):
    TEMP_IND = 1
    lines = []
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".out.dg_01", "r") as out_dg_fn:
        lines = out_dg_fn.readlines()
    DEBUG("SPC output:")
    for line in lines:
        DEBUG(line)

    #line = __pick_line_by_num_clusters(lines)
    lower_threshold = 25.0 # Disregard clusters smaller than this
    upper_threshold = min(non_masked_data_points * 0.99, non_masked_data_points - 1) # 99%
    if cluster_limit > 0:
        upper_threshold = min(upper_threshold, cluster_limit - 1)
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
    return hash( data.tostring() + str(dist_matrix) )

def __get_precalculated_spc_file_if_exists_for_data(data, dist_matrix):
    h = __calculate_hash_for_data(data, dist_matrix)
    return __get_precalculated_spc_file_if_exists(h)

def __save_calculated_spc_file_and_hash_for_data(data, dist_matrix, cluster, log):
    if not ALLOW_CACHING:
        return
    DEBUG("Saving calculated SPC cluster to file...")
    h = __calculate_hash_for_data(data, dist_matrix)
    save_to_file((cluster, log), SPC_CLUSTER_FILE.format(h))

def cluster_rows_spc(data, dist_matrix, cluster_limit):
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
    DEBUG("Preparing files for SPC run...")
    DEBUG("Preparing dat file...")
    n_data_points = __spc_prepare_dat_file(dist_matrix)
    DEBUG("Preparing edge file...")
    __spc_prepare_edge_file(n_data_points)
    DEBUG("Preparing run file...")
    __spc_prepare_run_file(n_data_points)
    DEBUG("Starting Super-Paramagnetic clustering...")
    __spc_run_and_wait_for_completion()
    non_masked_data_points = __spc_get_non_masked_data_points(dist_matrix)
    t, log = __spc_parse_temperature_results(non_masked_data_points, cluster_limit)
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

def cluster_rows(data, dist_matrix, cluster_limit=0):
    return cluster_rows_spc(data, dist_matrix, cluster_limit)

def __test_agglomerative_clustering():
    _, labels, ag = cluster_rows_agglomerative(None, sample_dist_matrix, 2)
    import ctwc__cluster_rank
    ranks_list = ctwc__cluster_rank.get_ranks(ag)
    ctwc__cluster_rank.get_nth_top_cluster_base_node(ranks_list)

def __test_dbscan_clustering(data, dist_matrix):
    _, labels, ag = cluster_rows_dbscan(None, dist_matrix)

def __test_spc_clustering():
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
    __test_spc_clustering()

def __inject_row_pattern_to_data(data):
    z = np.zeros(data.shape)
    for col in range(data.shape[1]): # Flooding to create a false relationship between rows
        for row in range(6):
            z[row, col] = (col % 2) * 200
        for row in range(data.shape[0] - 10, data.shape[0]):
            z[row, col] += (1 - (col % 2)) * 200
    return data + z

def __inject_col_pattern_to_data(data):
    z = np.zeros(data.shape)
    for col in range(30):
        for row in range(data.shape[0]): # Flooding to create a false relationship between rows
            z[row, col] = data[row, 0] - data[row, col]
    return data + z

if __name__ == "__main__":
    test()
