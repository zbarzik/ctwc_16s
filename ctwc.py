#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import ctwc__distance_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot, ctwc__metadata_analysis
import numpy as np

# Result tuple indices:
RES_IND_INPUT = 0
RES_IND_P_VAL = 1
RES_IND_DIST = 2
Q_VALUES_ITERATION_FILENAME = "q_vals_{0}.csv"

def __join_submatrices_by_axis(mat_a, mat_b, axis):
    ret = np.concatenate((mat_a, mat_b), axis)
    ASSERT(ret.shape[axis] == mat_a.shape[axis] + mat_b.shape[axis])
    return np.squeeze(ret)

def __join_submatrices_by_rows(mat_a, mat_b):
    return __join_submatrices_by_axis(mat_a, mat_b, 0)

def __sort_matrix_rows_by_selection(mat, selection):
    selection_compliment = list(set(range(mat.shape[0])) - set(selection))
    mat1 = mat[selection]
    mat2 = mat[selection_compliment]
    return __join_submatrices_by_rows(mat1, mat2)

def __sort_matrix_cols_by_selection(mat, selection):
    return __sort_matrix_rows_by_selection(mat.transpose(), selection).transpose()

"""
Filter OTUs by picked indices - mask out all entries EXCEPT the ones noted by the picked indices.
Compliment is from the previous filter (if provided).
All output is SORTED.
"""
def __prepare_otu_filters_from_indices(picked_indices, otus, prev_otu_filter = None):
    selected_rows_filter = [ otu for index, otu in enumerate(otus) if index in picked_indices ]
    compliment_rows_filter = [ otu for index, otu in enumerate(otus) if index not in picked_indices ]
    if prev_otu_filter is not None:
        compliment_rows_filter = [ otu for otu in compliment_rows_filter if otu in prev_otu_filter ]
    return sorted(selected_rows_filter), sorted(compliment_rows_filter)

"""
Filter samples by picked indices - mask out all entries EXCEPT the ones noted by the picked indices.
Compliment is from the previous filter (if provided).
All output is SORTED.
"""
def __prepare_sample_filters_from_indices(picked_indices, samples, prev_samp_filter = None):
    selected_cols_filter = [ samp for index, samp in enumerate(samples) if index in picked_indices ]
    compliment_cols_filter = [ samp for index, samp in enumerate(samples) if index not in picked_indices ]
    if prev_samp_filter is not None:
        compliment_cols_filter = [ samp for samp in compliment_cols_filter if samp in prev_samp_filter ]
    return sorted(selected_cols_filter), sorted(compliment_cols_filter)

__full_otus_dist = None
def __get_full_otus_dist(otus):
    if globals()['__full_otus_dist'] is None:
        full_otus_list, _ = __prepare_otu_filters_from_indices(range(len(otus)), otus)
        globals()['__full_otus_dist'] = ctwc__metadata_analysis.calculate_otus_distribution(full_otus_list)
    return globals()['__full_otus_dist']

__full_samples_dist = None
def __get_full_sample_dist(samples):
    if globals()['__full_samples_dist'] is None:
        full_samples_list, _ = __prepare_sample_filters_from_indices(range(len(samples)), samples)
        globals()['__full_samples_dist'] = ctwc__metadata_analysis.calculate_samples_distribution(full_samples_list)
    return globals()['__full_samples_dist']

def get_top_p_val(p_vals):
    mn = 1.0
    tpl = (None, None)
    for k_1 in p_vals:
        for k_2 in p_vals[k_1]:
            if p_vals[k_1][k_2] < mn:
                mn = p_vals[k_1][k_2]
                tpl = (k_1, k_2)
    return tpl, mn

def run_iteration(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, is_rows, prev=0):
    INFO("{0}: {1}".format(title, desc))
    INFO("Input size: {0} {1}".format(len(otus) if rows_filter is None else len(rows_filter),
                                      len(samples) if cols_filter is None else len(cols_filter)))
    if is_rows:
        return __run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, prev)
    else:
        return __run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, prev)

def __run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, prev=0):
    rows_dist, _ = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               sample_filter=cols_filter,
                                                               otu_filter=rows_filter,
                                                               skip_cols=True)

    ctwc__plot.plot_mat(rows_dist, header="{0}: {1}".format(title, "OTUs Distance Matrix"))

    picked_indices, last_rank, _, _, _, _ = ctwc__cluster_rank.filter_rows_by_top_rank(data,
                                                                                       rows_dist,
                                                                                       prev,
                                                                                       otus)

    selected_rows_filter, compliment_rows_filter = __prepare_otu_filters_from_indices(picked_indices, otus, rows_filter)

    sorted_rows_mat = __sort_matrix_rows_by_selection(rows_dist, picked_indices)
    sorted_mat = __sort_matrix_cols_by_selection(sorted_rows_mat, picked_indices)

    ctwc__plot.plot_mat(sorted_mat, header="{0}: {1}".format(title, "OTUs Distance Matrix - sorted"))

    if table is not None:
        picked_otus = ctwc__data_handler.get_otus_by_indices(picked_indices, table)
        taxonomies = ctwc__metadata_analysis.get_taxonomies_for_otus(picked_otus)
        DEBUG("Picked OTUs:")
        for taxonomy in taxonomies:
            DEBUG(taxonomy)
        ref_dist = __get_full_otus_dist(otus)
        sel_dist = ctwc__metadata_analysis.calculate_otus_distribution(selected_rows_filter)
        p_vals = ctwc__metadata_analysis.calculate_otus_p_values(sel_dist, ref_dist)
        DEBUG("P Values: {0}".format(p_vals))
        keys, pv = get_top_p_val(p_vals)
        INFO("Top P Value: {0}, keys: {1} {2}".format(pv, keys[0], keys[1]))

    num_otus = len(selected_rows_filter)
    num_samples = len(samples) if cols_filter == None else len(cols_filter)

    return (num_otus, num_samples), selected_rows_filter, compliment_rows_filter, p_vals, sel_dist

def __run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, prev=0):
    _, cols_dist = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               otu_filter=rows_filter,
                                                               sample_filter=cols_filter,
                                                               skip_rows=True)

    ctwc__plot.plot_mat(cols_dist, header="{0}: {1}".format(title, "Samples Distance Matrix"))

    picked_indices, last_rank, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data,
                                                                                       cols_dist,
                                                                                       prev,
                                                                                       samples)

    selected_cols_filter, compliment_cols_filter = __prepare_sample_filters_from_indices(picked_indices, samples, cols_filter)

    sorted_rows_mat = __sort_matrix_rows_by_selection(cols_dist, picked_indices)
    sorted_mat = __sort_matrix_cols_by_selection(sorted_rows_mat, picked_indices)

    ctwc__plot.plot_mat(sorted_mat, header="{0}: {1}".format(title, "Samples Distance Matrix - sorted"))

    INFO("Selected {0} samples:".format(len(picked_indices)))
    DEBUG(picked_indices)
    if table is not None:
        picked_samples = ctwc__data_handler.get_samples_by_indices(picked_indices, table)
        DEBUG(picked_samples)
        dates = ctwc__metadata_analysis.get_collection_dates_for_samples(picked_samples)
        DEBUG("Collection dates for selected samples:")
        for row in dates:
            DEBUG(row)
        ref_dist = __get_full_sample_dist(samples)
        sel_dist = ctwc__metadata_analysis.calculate_samples_distribution(selected_cols_filter)
        p_vals = ctwc__metadata_analysis.calculate_samples_p_values(sel_dist, ref_dist)
        DEBUG("P Values: {0}".format(p_vals))
        keys, pv = get_top_p_val(p_vals)
        INFO("Top P Value: {0}, keys: {1} {2}".format(pv, keys[0], keys[1]))

    num_otus = len(otus) if rows_filter == None else len(rows_filter)
    num_samples = len(selected_cols_filter)

    return (num_otus, num_samples), selected_cols_filter, compliment_cols_filter, p_vals, sel_dist

def __ctwc_recursive__get_iteration_indices(iteration_ind):
    if iteration_ind == "0":
        return [ "1", "2", None, None ]
    else:
        return [ iteration_ind + ".{0}".format(ind) for ind in range(1,5) ]

def __ctwc_recursive__get_next_step(iteration_ind, step):
    return __ctwc_recursive__get_iteration_indices(iteration_ind)[step]

def ctwc_recursive_select(data, tree, samples, otus, table):
    ctwc__plot.init()
    iteration_results = dict()
    __ctwc_recursive_iteration(data, tree, samples, otus, table, iteration_results = iteration_results)

    import csv
    for elem in iteration_results:
        filename = Q_VALUES_ITERATION_FILENAME.format(elem)
        with open(filename, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            iteration_results[elem] = (iteration_results[elem][RES_IND_INPUT], ctwc__metadata_analysis.correct_p_vals(iteration_results[elem][RES_IND_P_VAL]), iteration_results[elem][RES_IND_DIST])
            for k in iteration_results[elem][RES_IND_P_VAL]:
                ctwc__metadata_analysis.save_q_values_to_csv_for_iteration(csv_writer, k, iteration_results[elem][RES_IND_P_VAL], iteration_results[elem][RES_IND_DIST])

    ctwc__plot.wait_for_user()
    return iteration_results

def __ctwc_recursive_iteration(data, tree, samples, otus, table,
                             otu_filter = None,
                             otu_compliment = None,
                             sample_filter = None,
                             sample_compliment = None,
                             iteration_ind = "0",
                             iteration_results = dict()):

    THRESH = 100
    do_otus = otu_filter is None or len(otu_filter) > THRESH
    do_samples = sample_filter is None or len(sample_filter) > THRESH
    do_otus_comp = otu_compliment is not None and len(otu_compliment) > THRESH
    do_samples_comp = sample_compliment is not None and len(sample_compliment) > THRESH

    step = __ctwc_recursive__get_next_step(iteration_ind, 0)

    if len(iteration_ind) > len("x.x"):
        do_otus = False
        do_samples = False
        do_samples_comp = False
        do_otus_comp = False

    if do_samples:
        result, step_samp_filter, step_samp_compliment, p_vals, sel_dist = run_iteration("Iteration {0}".format(step), "Pick samples...",
                                                                       data,
                                                                       tree,
                                                                       samples,
                                                                       otus,
                                                                       otu_filter,
                                                                       sample_filter,
                                                                       table,
                                                                       False,
                                                                       0 if sample_filter is None else len(sample_filter))
        iteration_results["Iteration {0}".format(step)] = (result, p_vals, sel_dist)
        if sample_filter is None or (len(step_samp_filter) > 0 and len(step_samp_filter) < len(sample_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       otu_filter,
                                       otu_compliment,
                                       step_samp_filter,
                                       step_samp_compliment,
                                       step,
                                       iteration_results)

    if do_otus:
        step = __ctwc_recursive__get_next_step(iteration_ind, 1)
        result, step_otu_filter, step_otu_compliment, p_vals, sel_dist = run_iteration("Iteration {0}".format(step), "Pick OTUs...",
                                                                     data,
                                                                     tree,
                                                                     samples,
                                                                     otus,
                                                                     otu_filter,
                                                                     sample_filter,
                                                                     table,
                                                                     True,
                                                                     0 if otu_filter is None else len(otu_filter))
        iteration_results["Iteration {0}".format(step)] = (result, p_vals, sel_dist)
        if otu_filter is None or (len(step_otu_filter) > 0 and len(step_otu_filter) < len(otu_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       step_otu_filter,
                                       step_otu_compliment,
                                       sample_filter,
                                       sample_compliment,
                                       step,
                                       iteration_results)

    if do_samples_comp:
        step = __ctwc_recursive__get_next_step(iteration_ind, 2)
        result, step_samp_filter, step_samp_compliment, p_vals, sel_dist = run_iteration("Iteration {0}".format(step), "Pick samples from compliment...",
                                                             data,
                                                             tree,
                                                             samples,
                                                             otus,
                                                             otu_filter,
                                                             sample_compliment,
                                                             table,
                                                             False)
        iteration_results["Iteration {0}".format(step)] = (result, p_vals, sel_dist)
        if sample_filter is None or (len(step_samp_filter) < len(sample_compliment) and len(step_samp_filter) > 0):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       otu_filter,
                                       otu_compliment,
                                       step_samp_filter,
                                       step_samp_compliment,
                                       step,
                                       iteration_results)

    if do_otus_comp:
        step = __ctwc_recursive__get_next_step(iteration_ind, 3)
        result, step_otu_filter, step_otu_compliment, p_vals, sel_dist = run_iteration("Iteration {0}".format(step), "Pick OTUs from compliment...",
                                                             data,
                                                             tree,
                                                             samples,
                                                             otus,
                                                             otu_compliment,
                                                             sample_filter,
                                                             table,
                                                             True)
        iteration_results["Iteration {0}".format(step)] = (result, p_vals, sel_dist)
        if otu_compliment is None or (len(step_otu_filter) < len(otu_compliment) and len(step_otu_filter) > 0):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       step_otu_filter,
                                       step_otu_compliment,
                                       sample_filter,
                                       sample_compliment,
                                       step,
                                       iteration_results)


def test():
    np.seterr(all="ignore")
    samples, otus, tree, data, table = ctwc__distance_matrix.get_data(use_real_data=False, full_set=False)

    output = ctwc_recursive_select(data, tree, samples, otus, table)
    INFO("Full data size: {0} X {1}".format(data.shape[0], data.shape[1]))
    for elem in output:
        keys, pv = get_top_p_val(output[elem][RES_IND_P_VAL])
        INFO("{0}: {1} X {2} - P Value {3} Keys {4}".format(elem, output[elem][RES_IND_INPUT][0], output[elem][RES_IND_INPUT][1], pv, keys))

if __name__ == "__main__":
    test()
