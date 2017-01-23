#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import ctwc__distance_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot, ctwc__metadata_analysis
import numpy as np

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
    for k_1 in p_vals.keys():
        for k_2 in p_vals[k_1].keys():
            if p_vals[k_1][k_2] < mn:
                mn = p_vals[k_1][k_2]
                tpl = (k_1, k_2)
    return tpl, mn

def run_iteration(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, is_rows):
    INFO("{0}: {1}".format(title, desc))
    INFO("Input size: {0} {1}".format(len(otus) if rows_filter is None else len(rows_filter),
                                      len(samples) if cols_filter is None else len(cols_filter)))
    if is_rows:
        return __run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table)
    else:
        return __run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table)

def __run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table):
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

    return (num_otus, num_samples), selected_rows_filter, compliment_rows_filter, p_vals

def __run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table):
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

    return (num_otus, num_samples), selected_cols_filter, compliment_cols_filter, p_vals

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

    for elem in iteration_results:
        iteration_results[elem][1] = ctwc__metadata_analysis.correct_p_vals(iteration_results[elem][1])

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
    do_otus = False
    do_samples = False

    if otu_filter is None or (len(otu_filter) > THRESH and len(otu_compliment) > THRESH):
        do_otus = True
    if sample_filter is None or (len(sample_filter) > THRESH and len(sample_compliment) > THRESH):
        do_samples = True

    step = __ctwc_recursive__get_next_step(iteration_ind, 0)

    if len(iteration_ind) > len("x.x.x.x"):
        do_otus = False
        do_samples = False

    if do_samples:
        result, step_samp_filter, step_samp_compliment, p_vals = run_iteration("Iteration {0}".format(step), "Pick samples...",
                                                                       data,
                                                                       tree,
                                                                       samples,
                                                                       otus,
                                                                       otu_filter,
                                                                       sample_filter,
                                                                       table,
                                                                       False)
        iteration_results["Iteration {0}".format(step)] = (result, p_vals)
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
        result, step_otu_filter, step_otu_compliment, p_vals = run_iteration("Iteration {0}".format(step), "Pick OTUs...",
                                                                     data,
                                                                     tree,
                                                                     samples,
                                                                     otus,
                                                                     otu_filter,
                                                                     sample_filter,
                                                                     table,
                                                                     True)
        iteration_results["Iteration {0}".format(step)] = (result, p_vals)
        if otu_filter is None or (len(step_otu_filter) > 0 and len(step_otu_filter) < len(otu_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       step_otu_filter,
                                       step_otu_compliment,
                                       sample_filter,
                                       sample_compliment,
                                       step,
                                       iteration_results)

    step = __ctwc_recursive__get_next_step(iteration_ind, 2)
    if step is not None and do_samples:
        result, step_samp_filter, step_samp_compliment, p_vals = run_iteration("Iteration {0}".format(step), "Pick samples from compliment...",
                                                             data,
                                                             tree,
                                                             samples,
                                                             otus,
                                                             otu_filter,
                                                             sample_compliment,
                                                             table,
                                                             False)
        iteration_results["Iteration {0}".format(step)] = (result, p_vals)
        if sample_filter is None or (len(step_samp_filter) < len(sample_compliment) and len(step_samp_filter) > 0):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       otu_filter,
                                       otu_compliment,
                                       step_samp_filter,
                                       step_samp_compliment,
                                       step,
                                       iteration_results)

    step = __ctwc_recursive__get_next_step(iteration_ind, 3)
    if step is not None and do_otus:
        result, step_otu_filter, step_otu_compliment, p_vals = run_iteration("Iteration {0}".format(step), "Pick OTUs from compliment...",
                                                             data,
                                                             tree,
                                                             samples,
                                                             otus,
                                                             otu_compliment,
                                                             sample_filter,
                                                             table,
                                                             True)
        iteration_results["Iteration {0}".format(step)] = (result, p_vals)
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
    samples, otus, tree, data, table = ctwc__distance_matrix.get_data(use_real_data=True, full_set=True)

    output = ctwc_recursive_select(data, tree, samples, otus, table)
    INFO("Full data size: {0} X {1}".format(data.shape[0], data.shape[1]))
    for elem in output:
        pv, keys = get_top_p_val(output[elem][1])
        INFO("{0}: {1} X {2} - P Value {3} Keys {4}".format(elem, output[elem][0][0], output[elem][0][1], pv, keys))

if __name__ == "__main__":
    test()
