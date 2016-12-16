#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import ctwc__distnace_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot
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

def __prepare_otu_filters_from_indices(picked_indices, otus, prev_otu_filter = []):
    selected_rows_filter = [ otu for index, otu in enumerate(otus) if index not in picked_indices ]
    compliment_rows_filter = [ otu for index, otu in enumerate(otus) if index in picked_indices ]
    if len(prev_otu_filter) > 0:
        selected_rows_filter = [ otu for otu in selected_rows_filter if otu not in prev_otu_filter ]
        compliment_rows_filter = [ otu for otu in compliment_rows_filter if otu not in prev_otu_filter ]
    return selected_rows_filter, compliment_rows_filter

def __prepare_sample_filters_from_indices(picked_indices, samples, prev_samp_filter = []):
    selected_cols_filter = [ samp for index, samp in enumerate(samples) if index not in picked_indices ]
    compliment_cols_filter = [ samp for index, samp in enumerate(samples) if index in picked_indices ]
    if len(prev_samp_filter) > 0:
        selected_cols_filter = [ samp for samp in selected_cols_filter if samp in prev_samp_filter ]
        compliment_cols_filter = [ samp for samp in compliment_cols_filter if samp in prev_samp_filter ]
    return selected_cols_filter, compliment_cols_filter

def run_iteration(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, is_rows):
    INFO("{0}: {1}".format(title, desc))
    INFO("Input size: {0} {1}".format(len(otus) - len(rows_filter), len(samples) - len(cols_filter)))
    if is_rows:
        return __run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table)
    else:
        return __run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table)

def __run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table):
    rows_dist, _ = ctwc__distnace_matrix.get_distance_matrices(data,
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
        taxonomies = ctwc__data_handler.get_taxonomies_for_otus(picked_otus)
        INFO("Picked OTUs:")
        for taxonomy in taxonomies:
            INFO(taxonomy)

    num_otus = len(picked_indices)
    num_samples = len(samples) - len(cols_filter)

    return (num_otus, num_samples), selected_rows_filter, compliment_rows_filter

def __run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table):
    _, cols_dist = ctwc__distnace_matrix.get_distance_matrices(data,
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
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    num_otus = len(otus) - len(rows_filter)
    num_samples = len(picked_indices)

    return (num_otus, num_samples), selected_cols_filter, compliment_cols_filter

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

    ctwc__plot.wait_for_user()
    return iteration_results

def __ctwc_recursive_iteration(data, tree, samples, otus, table,
                             otu_filter = [],
                             otu_compliment = [],
                             sample_filter = [],
                             sample_compliment = [],
                             iteration_ind = "0",
                             iteration_results = dict()):

    THRESH = 100
    do_otus = False
    do_samples = False

    if len(otu_filter) == 0 or (len(otu_filter) > THRESH and len(otu_compliment) > THRESH):
        do_otus = True
    if len(sample_filter) == 0 or (len(sample_filter) > THRESH and len(sample_compliment) > THRESH):
        do_samples = True

    step = __ctwc_recursive__get_next_step(iteration_ind, 0)

    if len(iteration_ind) > len("x.x.x.x"):
        do_otus = False
        do_samples = False

    if do_samples:
        result, step_samp_filter, step_samp_compliment = run_iteration("Iteration {0}".format(step), "Pick samples...",
                                                                       data,
                                                                       tree,
                                                                       samples,
                                                                       otus,
                                                                       otu_filter,
                                                                       sample_filter,
                                                                       table,
                                                                       False)
        iteration_results["Iteration {0}".format(step)] = result
        if len(sample_filter) == 0 or (len(step_samp_filter) > 0 and len(step_samp_filter) < len(sample_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                      otu_filter,
                                      otu_compliment,
                                      step_samp_filter,
                                      step_samp_compliment,
                                      step,
                                      iteration_results)

    if do_otus:
        step = __ctwc_recursive__get_next_step(iteration_ind, 1)
        result, step_otu_filter, step_otu_compliment = run_iteration("Iteration {0}".format(step), "Pick OTUs...",
                                                                     data,
                                                                     tree,
                                                                     samples,
                                                                     otus,
                                                                     otu_filter,
                                                                     sample_filter,
                                                                     table,
                                                                     True)
        iteration_results["Iteration {0}".format(step)] = result
        if len(otu_filter) == 0 or (len(step_otu_filter) > 0 and len(step_otu_filter) < len(otu_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                      step_otu_filter,
                                      step_otu_compliment,
                                      sample_filter,
                                      sample_compliment,
                                      step,
                                      iteration_results)

    step = __ctwc_recursive__get_next_step(iteration_ind, 2)
    if step is not None and do_samples:
        result, step_samp_filter, step_samp_compliment = run_iteration("Iteration {0}".format(step), "Pick samples from compliment...",
                                                             data,
                                                             tree,
                                                             samples,
                                                             otus,
                                                             otu_filter,
                                                             sample_compliment,
                                                             table,
                                                             False)
        iteration_results["Iteration {0}".format(step)] = result
        if len(step_samp_filter) < len(sample_compliment) and len(step_samp_filter) > 0:
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                      otu_filter,
                                      otu_compliment,
                                      step_samp_filter,
                                      step_samp_compliment,
                                      step,
                                      iteration_results)

    step = __ctwc_recursive__get_next_step(iteration_ind, 3)
    if step is not None and do_otus:
        result, step_otu_filter, step_otu_compliment = run_iteration("Iteration {0}".format(step), "Pick OTUs from compliment...",
                                                             data,
                                                             tree,
                                                             samples,
                                                             otus,
                                                             otu_compliment,
                                                             sample_filter,
                                                             table,
                                                             True)
        iteration_results["Iteration {0}".format(step)] = result
        if len(step_otu_filter) < len(otu_compliment) and len(step_otu_filter) > 0:
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                      step_otu_filter,
                                      step_otu_compliment,
                                      sample_filter,
                                      sample_compliment,
                                      step,
                                      iteration_results)


def ctwc_select(data, tree, samples, otus, table):
    ctwc__plot.init()

    iteration_results = dict()
    result, samp_filter, samp_compliment = run_iteration("Iteration 1", "Pick samples from full dataset...",
                                                         data,
                                                         tree,
                                                         samples,
                                                         otus,
                                                         [],
                                                         [],
                                                         table,
                                                         False)
    iteration_results["Iteration 1"] = result

    result, otu_filter, otu_compliment = run_iteration("Iteration 1.1", "Pick OTUs from selected samples...",
                                                         data,
                                                         tree,
                                                         samples,
                                                         otus,
                                                         [],
                                                         samp_filter,
                                                         table,
                                                         True)
    iteration_results["Iteration 1.1"] = result

    result, otu_filter, otu_compliment = run_iteration("Iteration 1.2", "Pick OTUs from compliment of selected samples...",
                                                         data,
                                                         tree,
                                                         samples,
                                                         otus,
                                                         [],
                                                         samp_compliment,
                                                         table,
                                                         True)
    iteration_results["Iteration 1.2"] = result

    result, otu_filter, otu_compliment = run_iteration("Iteration 2", "Pick OTUs from full dataset...",
                                                         data,
                                                         tree,
                                                         samples,
                                                         otus,
                                                         [],
                                                         [],
                                                         table,
                                                         True)
    iteration_results["Iteration 2"] = result

    result, samp_filter, samp_compliment = run_iteration("Iteration 2.1", "Pick samples from selected OTUs...",
                                                         data,
                                                         tree,
                                                         samples,
                                                         otus,
                                                         otu_filter,
                                                         [],
                                                         table,
                                                         False)
    iteration_results["Iteration 2.1"] = result

    result, samp_filter, samp_compliment = run_iteration("Iteration 2.2", "Pick samples from compliment of selected OTUs...",
                                                         data,
                                                         tree,
                                                         samples,
                                                         otus,
                                                         otu_compliment,
                                                         [],
                                                         table,
                                                         False)
    iteration_results["Iteration 2.2"] = result

    ctwc__plot.wait_for_user()

    return iteration_results

def test():
    np.seterr(all="ignore")
    samples, otus, tree, data, table = ctwc__distnace_matrix.get_data(True)

    output = ctwc_recursive_select(data, tree, samples, otus, table)
    #output = ctwc_select(data, tree, samples, otus, table)
    INFO("Full data size: {0} X {1}".format(data.shape[0], data.shape[1]))
    for elem in output:
        INFO("{0}: {1} X {2}".format(elem, output[elem][0], output[elem][1]))

if __name__ == "__main__":
    test()
