#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP

import ctwc__distnace_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot
import numpy as np

def _join_submatrices_by_axis(mat_a, mat_b, axis):
    ret = np.concatenate((mat_a, mat_b), axis)
    ASSERT(ret.shape[axis] == mat_a.shape[axis] + mat_b.shape[axis])
    return np.squeeze(ret)

def _join_submatrices_by_rows(mat_a, mat_b):
    return _join_submatrices_by_axis(mat_a, mat_b, 0)

def _sort_matrix_rows_by_selection(mat, selection):
    selection_compliment = list(set(range(mat.shape[0])) - set(selection))
    mat1 = mat[selection]
    mat2 = mat[selection_compliment]
    return _join_submatrices_by_rows(mat1, mat2)

def _sort_matrix_cols_by_selection(mat, selection):
    return _sort_matrix_rows_by_selection(mat.transpose(), selection).transpose()

def prepare_otu_filters_from_indices(picked_indices, otus):
    selected_rows_filter = [ otu for index, otu in enumerate(otus) if index not in picked_indices ]
    compliment_rows_filter = [ otu for index, otu in enumerate(otus) if index in picked_indices ]
    return selected_rows_filter, compliment_rows_filter

def prepare_sample_filters_from_indices(picked_indices, samples):
    selected_cols_filter = [ samp for index, samp in enumerate(samples) if index not in picked_indices ]
    compliment_cols_filter = [ samp for index, samp in enumerate(samples) if index in picked_indices ]
    return selected_cols_filter, compliment_cols_filter

def run_iteration(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table, is_rows):
    if is_rows:
        return run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table)
    else:
        return run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table)

def run_iteration__rows(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table):
    INFO("{0}: {1}".format(title, desc))
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

    selected_rows_filter, compliment_rows_filter = prepare_otu_filters_from_indices(picked_indices, otus)

    sorted_rows_mat = _sort_matrix_rows_by_selection(rows_dist, picked_indices)

    ctwc__plot.plot_mat(sorted_rows_mat, header="{0}: {1}".format(title, "OTUs Distance Matrix - sorted"))

    if table is not None:
        picked_otus = ctwc__data_handler.get_otus_by_indices(picked_indices, table)
        taxonomies = ctwc__data_handler.get_taxonomies_for_otus(picked_otus)
        INFO("Picked OTUs:")
        for taxonomy in taxonomies:
            INFO(taxonomy)

    num_otus = len(picked_indices)
    num_samples = len(samples) - len(cols_filter)

    return (num_otus, num_samples), selected_rows_filter, compliment_rows_filter

def run_iteration__cols(title, desc, data, tree, samples, otus, rows_filter, cols_filter, table):
    INFO("{0}: {1}".format(title, desc))
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

    selected_cols_filter, compliment_cols_filter = prepare_sample_filters_from_indices(picked_indices, samples)

    sorted_cols_mat = _sort_matrix_rows_by_selection(cols_dist, picked_indices)

    ctwc__plot.plot_mat(sorted_cols_mat, header="{0}: {1}".format(title, "Samples Distance Matrix - sorted"))

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

def ctwc_select(data, tree, samples, otus, table):
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

    return iteration_results

def __ctwc_select__obsolete(data, tree, samples, otus, table):
    INFO("Iteration 1: Picking samples based on unifrac distance...")
    _, cols_dist_1 = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus, skip_rows=True)
    picked_indices_1, last_rank_1, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data, cols_dist_1, samples)
    selected_cols_filter_1, compliment_cols_filter_1 = prepare_sample_filters_from_indices(picked_indices_1, samples)

    INFO("Selected {0} samples:".format(len(picked_indices_1)))
    DEBUG(picked_indices_1)
    if table is not None:
        picked_samples_1 = ctwc__data_handler.get_samples_by_indices(picked_indices_1, table)
        DEBUG(picked_samples_1)
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples_1)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    INFO("Iteration 2: Re-picking samples after filtering out the cluster picked in Iteration 1...")
    _, cols_dist_2 = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                  sample_filter=compliment_cols_filter_1,
                                                                  skip_rows=True)
    picked_indices_2, last_rank_2, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data, cols_dist_2, samples)
    selected_cols_filter_2, compliment_cols_filter_2 = prepare_sample_filters_from_indices(picked_indices_2, samples)

    if table is not None:
        INFO("Selected {0} samples:".format(len(picked_indices_2)))
        picked_samples_2 = ctwc__data_handler.get_samples_by_indices(picked_indices_2, table)
        DEBUG(picked_samples_2)
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples_2)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    INFO("Iteration 1.1: Picking OTUs from selected samples...")
    rows_dist_1_1, _ = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                    sample_filter=selected_cols_filter_1,
                                                                    skip_cols=True)
    picked_indices_1_1, last_rank_1_1, _, _, _, _ = ctwc__cluster_rank.filter_rows_by_top_rank(data, rows_dist_1_1, otus)
    selected_rows_filter_1_1, compliment_rows_filter_1_1 = prepare_otu_filters_from_indices(picked_indices_1_1, otus)

    if table is not None:
        picked_otus_1_1 = ctwc__data_handler.get_otus_by_indices(picked_indices_1_1, table)
        taxonomies = ctwc__data_handler.get_taxonomies_for_otus(picked_otus_1_1)
        INFO("Picked OTUs:")
        for taxonomy in taxonomies:
            INFO(taxonomy)

    INFO("Iteration 1.2: Picking OTUs from selected samples compliment...")
    rows_dist_1_2, _ = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                    sample_filter=compliment_cols_filter_1,
                                                                    skip_cols=True)
    picked_indices_1_2, last_rank_1_2, _, _, _, _ = ctwc__cluster_rank.filter_rows_by_top_rank(data, rows_dist_1_2, otus)
    selected_rows_filter_1_2, compliment_rows_filter_1_2 = prepare_otu_filters_from_indices(picked_indices_1_2, otus)

    if table is not None:
        picked_otus_1_2 = ctwc__data_handler.get_otus_by_indices(picked_indices_1_2, table)
        taxonomies = ctwc__data_handler.get_taxonomies_for_otus(picked_otus_1_2)
        INFO("Picked OTUs:")
        for taxonomy in taxonomies:
            INFO(taxonomy)

    INFO("Iteration 3: Re-picking samples based on the compliment for OTUs in step 1.1...")
    selected_rows_filter_1_1, compliment_rows_filter_1_1 = prepare_otu_filters_from_indices(picked_indices_1_1, otus)
    _, cols_dist_3 = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                  otu_filter=compliment_rows_filter_1_1,
                                                                  skip_rows=True)
    picked_indices_3, last_rank_3, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data, cols_dist_3, samples)
    selected_cols_filter_3, compliment_cols_filter_3 = prepare_sample_filters_from_indices(picked_indices_3, samples)

    if table is not None:
        picked_samples_3 = ctwc__data_handler.get_samples_by_indices(picked_indices_3, table)
        DEBUG(picked_samples_3)
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples_3)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    INFO("Iteration 4: Re-picking samples based on the OTUs picked in step 1.1...")
    _, cols_dist_4 = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                  otu_filter=selected_rows_filter_1_1,
                                                                  skip_rows=True)
    picked_indices_4, last_rank_4, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data, cols_dist_4, samples)
    selected_cols_filter_4, compliment_cols_filter_4 = prepare_sample_filters_from_indices(picked_indices_4, samples)

    if table is not None:
        picked_samples_4 = ctwc__data_handler.get_samples_by_indices(picked_indices_4, table)
        DEBUG(picked_samples_4)
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples_4)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    INFO("Iteration 5: Picking OTUs from full dataset...")
    rows_dist_5, _ = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus, skip_cols=True)
    picked_indices_5, last_rank_5, _, _, _, _ = ctwc__cluster_rank.filter_rows_by_top_rank(data, rows_dist_5, otus)
    selected_rows_filter_5, compliment_rows_filter_5 = prepare_otu_filters_from_indices(picked_indices_5, otus)

    if table is not None:
        picked_otus_5 = ctwc__data_handler.get_otus_by_indices(picked_indices_5, table)
        taxonomies = ctwc__data_handler.get_taxonomies_for_otus(picked_otus_5)
        INFO("Picked OTUs:")
        for taxonomy in taxonomies:
            INFO(taxonomy)

    INFO("Iteration 5.1: Picking samples based on selected OTUs...")
    _, cols_dist_5_1 = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                    otu_filter=selected_rows_filter_5,
                                                                    skip_rows=True)
    picked_indices_5_1, last_rank_5_1, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data, cols_dist_5_1, samples)
    selected_cols_filter_5_1, compliment_cols_filter_5_1 = prepare_sample_filters_from_indices(picked_indices_5_1, samples)

    if table is not None:
        picked_samples_5_1 = ctwc__data_handler.get_samples_by_indices(picked_indices_5_1, table)
        DEBUG(picked_samples_5_1)
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples_5_1)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    INFO("Iteration 5.2: Picking samples based on selected OTUs compliment...")
    _, cols_dist_5_2 = ctwc__distnace_matrix.get_distance_matrices(data, tree, samples, otus,
                                                                    otu_filter=compliment_rows_filter_5,
                                                                    skip_rows=True)
    picked_indices_5_2, last_rank_5_2, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data, cols_dist_5_2, samples)
    selected_cols_filter_5_2, compliment_cols_filter_5_2 = prepare_sample_filters_from_indices(picked_indices_5_2, samples)

    if table is not None:
        picked_samples_5_2 = ctwc__data_handler.get_samples_by_indices(picked_indices_5_2, table)
        DEBUG(picked_samples_5_2)
        dates = ctwc__data_handler.get_collection_dates_for_samples(picked_samples_5_2)
        INFO("Collection dates for selected samples:")
        for row in dates:
            INFO(row)

    output = { "Iteration 1"   : (len(otus), len(picked_indices_1)),
               "Iteration 2"   : (len(otus), len(picked_indices_2)),
               "Iteration 1.1" : (len(picked_indices_1_1), len(picked_indices_1)),
               "Iteration 1.2" : (len(picked_indices_1_2), len(selected_cols_filter_1)),
               "Iteration 3"   : (len(picked_indices_1_1), len(picked_indices_3)),
               "Iteration 4"   : (len(compliment_rows_filter_1_1), len(picked_indices_4)),
               "Iteration 5"   : (len(picked_indices_5), len(samples)),
               "Iteration 5.1" : (len(picked_indices_5), len(picked_indices_5_1)),
               "Iteration 5.2" : (len(compliment_rows_filter_5), len(picked_indices_5_2)),
               }
    return output

def test():
    np.seterr(all="ignore")
    samples, otus, tree, data, table = ctwc__distnace_matrix.get_data(True)

    output = ctwc_select(data, tree, samples, otus, table)
    INFO("Full data size: {0} X {1}".format(data.shape[0], data.shape[1]))
    for elem in output:
        INFO("{0}: {1} X {2}".format(elem, output[elem][0], output[elem][1]))

if __name__ == "__main__":
    test()
