#!/usr/bin/python
from ctwc__common import *
import warnings

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import ctwc__distance_matrix, ctwc__data_handler


# Constants
N_CLUSTERS = 16
MIN_DATA_POINTS_LIMIT = 50
MIN_CLUSTER_LIMIT = 20.0
SPC_BINARY_PATH = './SPC/'
SPC_BINARY_EXE = './SW'
SPC_TMP_FILES_PREFIX = '__tmp_ctwc'
SPC_CLUSTER_FILE = RESULTS_PATH+"spc_cluster-{0}.pklz"
ALLOW_CACHING = False
MIN_SCORE = 8
LARGEST_CLUSTER_IND = 4

# Simulates a distance matrix with two natural clusters. Expected result is (1,0,1,0,1).
sample_dist_matrix = np.array([ [ 0.0, 0.9, 0.1, 0.9, 0.1 ],
                                [ 0.9, 0.0, 0.9, 0.1, 0.9 ],
                                [ 0.1, 0.9, 0.0, 0.9, 0.1 ],
                                [ 0.1, 0.9, 0.9, 0.0, 0.9 ],
                                [ 0.1, 0.9, 0.1, 0.9, 0.0 ]
                                ])

__map_spc_to_mat = {}
__map_mat_to_spc = {}

def __spc_prepare_run_file(n_data_points):
    run_file = """
NumberOfPoints: {0}
NumberOfEdges: {2}
DataFile: {1}.dat
MinTemp: 0.0
MaxTemp: 0.5
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

def __is_row_filtered(dist_mat, row):
    non_filtered_indices = dist_mat[row] >= 1 # == ctwc__distance_matrix.INF_VALUE
    non_filtered_indices[row] = True # diagonal is zeros even for filtered, remove those.
    return non_filtered_indices.all()

def __spc_prepare_dat_file(dist_mat):
    globals()['__map_spc_to_mat'] = {}
    globals()['__map_mat_to_spc'] = {}
    lines = []
    count = 0
    for r in xrange(dist_mat.shape[0]):
        if __is_row_filtered(dist_mat, r):
            continue
        count += 1
        __map_spc_to_mat[count] = r
        __map_mat_to_spc[r] = count
    for r in __map_spc_to_mat.keys():
        for c in __map_spc_to_mat.keys():
            lines.append("{0} {1} {2}\n".format(r, c, dist_mat[__map_spc_to_mat[r]][__map_spc_to_mat[c]]))
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".dat", 'w+') as fn:
        for line in lines:
            fn.write(line)
    return count

def __spc_get_non_masked_data_points(dist_mat):
    n = dist_mat.shape[0]
    tmp = np.zeros(dist_mat.shape)
    tmp[dist_mat < ctwc__distance_matrix.INF_VALUE] = 1
    np.fill_diagonal(tmp, 0.0)
    non_zero_rows = len(set(tmp.nonzero()[0]))
    INFO("Non-masked rows: {0}".format(non_zero_rows))
    return non_zero_rows

def __spc_prepare_edge_file(dist_mat):
    n = dist_mat.shape[0]
    lines = []
    DEBUG("Size of __map_mat_to_spc is: {0}".format(len(__map_mat_to_spc)))
    for r in __map_mat_to_spc.keys():
        for c in __map_mat_to_spc.keys():
            lines.append("{0} {1}\n".format(r, c))

    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".edge", "w+") as edge_f:
        for line in lines:
            edge_f.write(line)

def __spc_run_and_wait_for_completion():
    from subprocess import call
    args = [SPC_BINARY_EXE, SPC_TMP_FILES_PREFIX + ".run"]
    call(args, cwd=SPC_BINARY_PATH)

def __check_clustering_success(lines):
    # excplicit handling of edge case: if there are no linked components
    # for SPC's k-means stage, everything breaks down immediately.
    return 1 != int(lines[1].split()[LARGEST_CLUSTER_IND])

def __pick_line_by_most_stable_largest_cluster(lines, lower_threshold=0, upper_threshold=float('Inf')):
    NUM_COLUMNS_TO_CHECK = 3
    if not __check_clustering_success(lines):
        return None, None
    top_column = -1
    top_score = -1
    top_line = None
    top_size = -1
    def is_valid_size(size):
        return size < upper_threshold and size > lower_threshold
    for column in xrange(LARGEST_CLUSTER_IND, LARGEST_CLUSTER_IND + NUM_COLUMNS_TO_CHECK):
        tmp_line, tmp_score = __pick_line_by_most_stable_largest_cluster_for_column(column, lines,  lower_threshold,  upper_threshold)
        tmp_size = int(tmp_line.split()[column])
        if (top_line == None or # first iteration
            tmp_score > MIN_SCORE and is_valid_size(tmp_size) and tmp_size > top_size or
            tmp_score > MIN_SCORE and tmp_size > top_size):
            top_column = column
            top_line = tmp_line
            top_score = tmp_score
            top_size = tmp_size
    return top_line, top_column - LARGEST_CLUSTER_IND

def __pick_line_by_most_stable_largest_cluster_for_column(column, lines, lower_threshold=0, upper_threshold=float('Inf')):
    from itertools import groupby
    def get_count_per_value(iterator):
        return sum(1 for _ in iterator)

    def get_counter_list(iterable):
        return [ (k, get_count_per_value(g)) for k, g in groupby(iterable) ]

    def get_max_counter(counter_list, lower_threshold):
        return max(counter_list, key=lambda item: item[1] if item[0] > lower_threshold and item[0] < upper_threshold and item[1] > MIN_SCORE else -1)

    def get_line_num_for_max_counter(counter_list, max_counter):
        ind = 0
        for counter in counter_list:
            if counter == max_counter:
                break
            else:
                ind += counter[1]
        return ind

    cluster_list = [ int(line.split()[column]) for line in lines ]
    counter_list = get_counter_list(cluster_list)
    max_counter = get_max_counter(counter_list, lower_threshold)
    while (max_counter[1] < MIN_SCORE and lower_threshold > 1
          or max_counter[0] == cluster_list[0] and len(counter_list) > 1):
        lower_threshold /= 2.0
        max_counter = get_max_counter(counter_list, lower_threshold)
    ind = get_line_num_for_max_counter(counter_list, max_counter)
    return lines[ind], max_counter[1]

def __spc_parse_temperature_results(non_masked_data_points, cluster_limit):
    TEMP_IND = 1
    lines = []
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".out.dg_01", "r") as out_dg_fn:
        lines = out_dg_fn.readlines()
    DEBUG("SPC output:")
    for line in lines:
        DEBUG(line)

    lower_threshold = MIN_CLUSTER_LIMIT # Disregard clusters smaller than this
    upper_threshold = min(non_masked_data_points * 0.99, non_masked_data_points - lower_threshold) # 99%
    if cluster_limit > 0:
        upper_threshold = min(upper_threshold, cluster_limit - 1)
    line, clust_id = __pick_line_by_most_stable_largest_cluster(lines, lower_threshold, upper_threshold)
    if line is None:
        return 0.0, lines, 0
    temperature = float(line.split()[TEMP_IND])

    INFO("Most stable temperature: {0}".format(temperature))
    return temperature, lines, clust_id

def __spc_get_clusters_by_temperature(t):
    TEMP_IND = 1
    with open(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + ".out.dg_01.lab", "r") as out_dg_lab_fn:
        lines = out_dg_lab_fn.readlines()
    for line in lines:
        if float(line.split()[TEMP_IND]) == t:
            break
    spc_indices = map(int, line.split()[TEMP_IND + 1:])
    return spc_indices

def __spc_get_cluster_members_by_cluster_id(clusters, cluster_id):
    clust = []
    for i, val in enumerate(clusters):
        if val == cluster_id:
            clust.append(__map_spc_to_mat[i + 1])
    return clust

def __spc_clear_temporary_files():
    from glob import glob
    import os
    map(os.remove, glob(SPC_BINARY_PATH + SPC_TMP_FILES_PREFIX + '*'))

def __get_precalculated_spc_file_if_exists(h):
    return load_from_file(SPC_CLUSTER_FILE.format(h))

def __calculate_hash_for_data(data, dist_matrix):
    if data is None or dist_matrix is None:
        return 0
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
    if n_data_points < MIN_DATA_POINTS_LIMIT:
        DEBUG("Too few data points, skipping clustering")
        return data, [0], None
    DEBUG("Preparing edge file...")
    __spc_prepare_edge_file(dist_matrix)
    DEBUG("Preparing run file...")
    __spc_prepare_run_file(n_data_points)
    DEBUG("Starting Super-Paramagnetic clustering...")
    __spc_run_and_wait_for_completion()
    non_masked_data_points = __spc_get_non_masked_data_points(dist_matrix)
    t, log, clust_id = __spc_parse_temperature_results(non_masked_data_points, cluster_limit)
    if t == 0.0:
        top_cluster = __spc_get_cluster_members_by_cluster_id([ 0 ] * n_data_points, 0)
    else:
        clusters = __spc_get_clusters_by_temperature(t)
        top_cluster = __spc_get_cluster_members_by_cluster_id(clusters, clust_id)
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
    SIZE = 101
    z = np.zeros((SIZE, SIZE))
    for i in range(0, SIZE):
        for j in range(0, SIZE):
            z[i][j] = 0.1 if (i+j) % 3 == 0 else 0.8
            if i == j:
                z[i][j] = 0.0
    INFO(z)
    _, top_cluster, _ = cluster_rows_spc(None, z, 50)
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
