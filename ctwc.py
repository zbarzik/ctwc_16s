#!/usr/bin/python
from ctwc__common import *

import ctwc__distance_matrix, ctwc__cluster_rank, ctwc__data_handler, ctwc__plot, ctwc__metadata_analysis
import numpy as np
import traceback

# Result tuple indices:
RES_IND_INPUT = 0
RES_IND_P_VAL = 1
RES_IND_SEL_DIST = 2
RES_IND_REF_DIST = 3
RES_IND_NUM_SELECTED = 4
RES_IND_NUM_TOTAL = 5

RECURSION_DEPTH = 3

Q_VALUES_ITERATION_FILENAME = RESULTS_PATH+"q_vals_{0}.csv"

RECURSIVE_THRESHOLD = 100

COUPLED_ENRICHMENT_THRESHOLD = 1.5

CLUSTER_OUTPUT_FILE = RESULTS_PATH+"cluster_results_{0}.txt"

BANNER_LEN = 50

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
    selected_otu_filter = [ otu for index, otu in enumerate(otus) if index in picked_indices ]
    compliment_otu_filter = [ otu for index, otu in enumerate(otus) if index not in picked_indices ]
    if prev_otu_filter is not None:
        compliment_otu_filter = [ otu for otu in compliment_otu_filter if otu in prev_otu_filter ]
    return sorted(selected_otu_filter), sorted(compliment_otu_filter)

"""
Filter samples by picked indices - mask out all entries EXCEPT the ones noted by the picked indices.
Compliment is from the previous filter (if provided).
All output is SORTED.
"""
def __prepare_sample_filters_from_indices(picked_indices, samples, prev_samp_filter = None):
    selected_samp_filter = [ samp for index, samp in enumerate(samples) if index in picked_indices ]
    compliment_samp_filter = [ samp for index, samp in enumerate(samples) if index not in picked_indices ]
    if prev_samp_filter is not None:
        compliment_samp_filter = [ samp for samp in compliment_samp_filter if samp in prev_samp_filter ]
    return sorted(selected_samp_filter), sorted(compliment_samp_filter)

__full_otus_dist = None
def __get_full_otus_dist(otus, table):
    if globals()['__full_otus_dist'] is None:
        full_otus_list, _ = __prepare_otu_filters_from_indices(range(len(otus)), otus)
        globals()['__full_otus_dist'] = ctwc__metadata_analysis.calculate_otus_distribution(full_otus_list, range(len(otus)), table)
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
        if k_1 in ctwc__metadata_analysis.OTU_RANKS_TO_SKIP:
            continue
        for k_2 in p_vals[k_1]:
            if p_vals[k_1][k_2] < mn:
                mn = p_vals[k_1][k_2]
                tpl = (k_1, k_2)
    return tpl, mn

def filter_p_vals_by_threshold(p_vals, t):
    filtered_p_vals = {}
    for k_1 in p_vals:
        for k_2 in p_vals[k_1]:
            p = p_vals[k_1][k_2]
            if p < t:
                if filtered_p_vals.has_key(k_1):
                    filtered_p_vals[k_1][k_2] = p
                else:
                    filtered_p_vals = {k_2: p}
    return filtered_p_vals

def run_iteration(title, desc, data, tree, samples, otus, otu_filter, sample_filter, table, is_rows, prev=0):
    INFO("{0}: {1}".format(title, desc))
    INFO("Input size: {0} {1}".format(len(otus) if otu_filter is None else len(otu_filter),
                                      len(samples) if sample_filter is None else len(sample_filter)))
    if is_rows:
        return __run_iteration__rows(title, desc, data, tree, samples, otus, otu_filter, sample_filter, table, prev)
    else:
        return __run_iteration__cols(title, desc, data, tree, samples, otus, otu_filter, sample_filter, table, prev)

def create_or_open_results_file(iteration):
    fd = open(CLUSTER_OUTPUT_FILE.format(make_camel_from_string(iteration)), 'a')
    return fd

def add_line_to_results_file(fd, line):
    if fd and not fd.closed:
        fd.write(line + "\n")

def get_samples_for_otus(data_in, picked_indices, otu_filter, sample_filter, table, samples, otus):
    data = np.copy(data_in)
    if otu_filter is not None:
        if otus is not list:
            otus = otus.tolist()
        rows_filter = [ otus.index(otu) for otu in otu_filter ]
        mask = np.ones(data.shape, dtype=bool)
        mask[rows_filter] = False
        data[mask] = 0.0
    if sample_filter is not None:
        if samples is not list:
            samples = samples.tolist()
        cols_filter = [ samples.index(samp) for samp in sample_filter ]
        mask = np.ones(data.shape, dtype=bool)
        mask[ :, cols_filter ] = False
        data[mask] = 0
    mask = np.ones(data.shape, dtype=bool)
    mask[picked_indices] = False
    data[mask] = 0
    samp_indices = numpy.where(data.any(axis=1))[0]
    return ctwc__data_handler.get_samples_by_indices(samp_indices, table), samp_indices

def get_otus_for_samples(data_in, picked_indices, otu_filter, sample_filter, table, samples, otus):
    data = np.copy(data_in)
    if otu_filter is not None:
        if otus is not list:
            otus = otus.tolist()
        rows_filter = [ otus.index(otu) for otu in otu_filter ]
        mask = np.ones(data.shape, dtype=bool)
        mask[rows_filter] = False
        data[mask] = 0.0
    if sample_filter is not None:
        if samples is not list:
            samples = samples.tolist()
        cols_filter = [ samples.index(samp) for samp in sample_filter ]
        mask = np.ones(data.shape, dtype=bool)
        mask[ :, cols_filter ] = False
        data[mask] = 0
    mask = np.ones(data.shape, dtype=bool)
    mask[ :, picked_indices ] = False
    data[mask] = 0
    otu_indices = numpy.where(data.any(axis=0))[0]
    return ctwc__data_handler.get_otus_by_indices(otu_indices, table), otu_indices

def get_enriched_keys_over_threshold(sel_dist, ref_dist, t):
    output = {}
    for field in sel_dist:
        for val in sel_dist[field][1]:
            sel_val = sel_dist[field][1][val]
            ref_val = ref_dist[field][1][val]
            ratio = sel_val / ref_val
            if ratio > t:
                output[val] = sel_val, ref_val, ratio
    return output

def __run_iteration__rows(title, desc, data, tree, samples, otus, otu_filter, sample_filter, table, prev=0):
    rows_dist, _ = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               sample_filter=sample_filter,
                                                               otu_filter=otu_filter,
                                                               skip_cols=True)

    picked_indices, last_rank, _, _, _, _ = ctwc__cluster_rank.filter_rows_by_top_rank(data,
                                                                                       rows_dist,
                                                                                       prev,
                                                                                       otus)

    selected_otu_filter, compliment_otu_filter = __prepare_otu_filters_from_indices(picked_indices, otus, otu_filter)

    res_file = create_or_open_results_file(title)
    add_line_to_results_file(res_file, "{0} - {1} OTUs X {2} Samples".format(title,
                                            len(otus) if otu_filter is None else len(otu_filter),
                                            len(samples) if sample_filter is None else len(sample_filter)))
    add_line_to_results_file(res_file, get_iteration_path_string(title_to_iteration(title)))
    add_line_to_results_file(res_file, "-"*BANNER_LEN)
    sorted_rows_mat = __sort_matrix_rows_by_selection(rows_dist, picked_indices)
    sorted_mat = __sort_matrix_cols_by_selection(sorted_rows_mat, picked_indices)

    ctwc__plot.plot_mat(sorted_mat, header="{0}: {1}".format(title, "OTUs Distance Matrix - sorted"))

    if table is not None:
        taxonomies = ctwc__metadata_analysis.get_taxa_by_otu_indices(picked_indices, table)
        picked_otus = ctwc__data_handler.get_otus_by_indices(picked_indices, table)
        INFO("Selected {0} OTUs".format(len(picked_indices)))
        add_line_to_results_file(res_file, "Selected {0} OTUs:".format(len(picked_indices)))
        add_line_to_results_file(res_file, "-"*BANNER_LEN)
        for taxonomy in taxonomies:
            DEBUG(taxonomy)
            add_line_to_results_file(res_file, taxonomy)

        add_line_to_results_file(res_file, "Selected OTU IDs:")
        add_line_to_results_file(res_file, "-"*BANNER_LEN)
        for picked_otu in picked_otus:
            DEBUG(picked_otu)
            add_line_to_results_file(res_file, picked_otu)

        add_line_to_results_file(res_file, "Included Samples:")
        add_line_to_results_file(res_file, "-"*BANNER_LEN)
        included_samples, included_samples_indices = get_samples_for_otus(
            data, picked_indices, otu_filter, sample_filter, table, samples, otus
        )
        for sample in included_samples:
            add_line_to_results_file(res_file, sample)

        ref_dist = __get_full_otus_dist(otus, table)
        sel_dist = ctwc__metadata_analysis.calculate_otus_distribution(selected_otu_filter, picked_indices, table)
        p_vals = ctwc__metadata_analysis.calculate_otus_p_values(sel_dist, ref_dist)
        DEBUG("P Values: {0}".format(p_vals))
        keys, pv = get_top_p_val(p_vals)
        INFO("Top P Value: {0}, keys: {1} {2}".format(pv, keys[0], keys[1]))

        coupled_sample_filter, _ = __prepare_sample_filters_from_indices(included_samples_indices, samples, sample_filter)
        if len(coupled_sample_filter) > 0:
            coupled_ref_dist = __get_full_sample_dist(samples)
            coupled_sel_dist = ctwc__metadata_analysis.calculate_samples_distribution(coupled_sample_filter)
            enriched = get_enriched_keys_over_threshold(coupled_sel_dist, coupled_ref_dist, COUPLED_ENRICHMENT_THRESHOLD)
            if len(enriched) > 0:
                add_line_to_results_file(res_file, "Enriched Samples:")
                add_line_to_results_file(res_file, "-"*BANNER_LEN)
            for key in enriched:
                add_line_to_results_file(res_file, "{0}: Selection: {1} Reference: {2} Enrichment: {3}".format(
                    key, enriched[key][0], enriched[key][1], enriched[key][2])
                )


    num_otus = len(selected_otu_filter)
    num_samples = len(samples) if sample_filter == None else len(sample_filter)

    res_file.close()

    return (num_otus, num_samples), selected_otu_filter, compliment_otu_filter, p_vals, (sel_dist, ref_dist)

def __run_iteration__cols(title, desc, data, tree, samples, otus, otu_filter, sample_filter, table, prev=0):
    _, cols_dist = ctwc__distance_matrix.get_distance_matrices(data,
                                                               tree,
                                                               samples,
                                                               otus,
                                                               otu_filter=otu_filter,
                                                               sample_filter=sample_filter,
                                                               skip_rows=True)

    picked_indices, last_rank, _, _, _, _ = ctwc__cluster_rank.filter_cols_by_top_rank(data,
                                                                                       cols_dist,
                                                                                       prev,
                                                                                       samples)

    selected_sample_filter, compliment_sample_filter = __prepare_sample_filters_from_indices(picked_indices, samples, sample_filter)

    res_file = create_or_open_results_file(title)
    add_line_to_results_file(res_file, "{0} - {1} OTUs X {2} Samples".format(title,
                                            len(otus) if otu_filter is None else len(otu_filter),
                                            len(samples) if sample_filter is None else len(sample_filter)))

    add_line_to_results_file(res_file, get_iteration_path_string(title_to_iteration(title)))
    add_line_to_results_file(res_file, "-"*BANNER_LEN)
    sorted_rows_mat = __sort_matrix_rows_by_selection(cols_dist, picked_indices)
    sorted_mat = __sort_matrix_cols_by_selection(sorted_rows_mat, picked_indices)

    ctwc__plot.plot_mat(sorted_mat, header="{0}: {1}".format(title, "Samples Distance Matrix - sorted"))

    if table is not None:
        INFO("Selected {0} samples".format(len(picked_indices)))
        add_line_to_results_file(res_file, "Selected {0} samples:".format(len(picked_indices)))
        add_line_to_results_file(res_file, "-"*BANNER_LEN)
        picked_samples = ctwc__data_handler.get_samples_by_indices(picked_indices, table)
        for samp in picked_samples:
            add_line_to_results_file(res_file, samp)
            DEBUG(samp)
        ref_dist = __get_full_sample_dist(samples)
        sel_dist = ctwc__metadata_analysis.calculate_samples_distribution(selected_sample_filter)
        p_vals = ctwc__metadata_analysis.calculate_samples_p_values(sel_dist, ref_dist)
        DEBUG("P Values: {0}".format(p_vals))
        keys, pv = get_top_p_val(p_vals)
        INFO("Top P Value: {0}, keys: {1} {2}".format(pv, keys[0], keys[1]))
        included_otus, included_otus_indices = get_otus_for_samples(
            data, picked_indices, otu_filter, sample_filter, table, samples, otus
        )
        coupled_otu_filter, _ = __prepare_otu_filters_from_indices(included_otus_indices, otus, otu_filter)
        add_line_to_results_file(res_file, "Included OTUs:")
        add_line_to_results_file(res_file, "-"*BANNER_LEN)
        for otu in included_otus:
            add_line_to_results_file(res_file, otu)
        if len(coupled_otu_filter) > 0:
            coupled_ref_dist = __get_full_otus_dist(otus, table)
            coupled_sel_dist = ctwc__metadata_analysis.calculate_otus_distribution(
                coupled_otu_filter, included_otus_indices, table
            )
            enriched = get_enriched_keys_over_threshold(coupled_sel_dist, coupled_ref_dist, COUPLED_ENRICHMENT_THRESHOLD)
            if len(enriched) > 0:
                add_line_to_results_file(res_file, "Enriched OTUs:")
                add_line_to_results_file(res_file, "-"*BANNER_LEN)
            for key in enriched:
                add_line_to_results_file(res_file, "{0}: Selection: {1} Reference: {2} Enrichment: {3}".format(
                    key, enriched[key][0], enriched[key][1], enriched[key][2])
                )

    num_otus = len(otus) if otu_filter == None else len(otu_filter)
    num_samples = len(selected_sample_filter)

    res_file.close()

    return (num_otus, num_samples), selected_sample_filter, compliment_sample_filter, p_vals, (sel_dist, ref_dist)

def __ctwc_recursive__get_iteration_indices(iteration_ind):
    if iteration_ind == "0":
        return [ "1", "2", None, None ]
    else:
        return [ iteration_ind + ".{0}".format(ind) for ind in xrange(1,7) ]

def __ctwc_recursive__get_next_step(iteration_ind, step):
    return __ctwc_recursive__get_iteration_indices(iteration_ind)[step]

def ctwc_recursive_select(data, tree, samples, otus, table):
    iteration_results = dict()
    __ctwc_recursive_iteration(data, tree, samples, otus, table, iteration_results = iteration_results)

    import csv
    for elem in iteration_results:
        filename = Q_VALUES_ITERATION_FILENAME.format(make_camel_from_string(elem))

        with open(filename, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            iteration_results[elem] = (iteration_results[elem][RES_IND_INPUT],
                                       ctwc__metadata_analysis.correct_p_vals(iteration_results[elem][RES_IND_P_VAL]),
                                       iteration_results[elem][RES_IND_SEL_DIST],
                                       iteration_results[elem][RES_IND_REF_DIST],
                                       iteration_results[elem][RES_IND_NUM_SELECTED],
                                       iteration_results[elem][RES_IND_NUM_TOTAL])
            for k in iteration_results[elem][RES_IND_P_VAL]:
                ctwc__metadata_analysis.save_q_values_to_csv_for_iteration(csv_writer,
                                                                           k,
                                                                           iteration_results[elem][RES_IND_P_VAL],
                                                                           iteration_results[elem][RES_IND_SEL_DIST],
                                                                           iteration_results[elem][RES_IND_REF_DIST],
                                                                           iteration_results[elem][RES_IND_NUM_SELECTED],
                                                                           iteration_results[elem][RES_IND_NUM_TOTAL])
        with open(filename, 'r') as q_vals:
            res_file = create_or_open_results_file(elem)
            lines = q_vals.readlines()
            add_line_to_results_file(res_file, "-"*BANNER_LEN)
            add_line_to_results_file(res_file, "Filtered Q values:")
            add_line_to_results_file(res_file, "-"*BANNER_LEN)
            for line in lines:
                add_line_to_results_file(res_file, line.strip())
            res_file.close()

    ctwc__plot.wait_for_user()
    return iteration_results

def __is_iteration_max_depth(iteration_ind):
    return len(iteration_ind) > len("x." * (RECURSION_DEPTH - 1)) - 1

def __ctwc_recursive_iteration(data, tree, samples, otus, table,
                             otu_filter = None,
                             otu_compliment = None,
                             sample_filter = None,
                             sample_compliment = None,
                             iteration_ind = "0",
                             iteration_results = dict()):

    if __is_iteration_max_depth(iteration_ind):
        return

    THRESH = RECURSIVE_THRESHOLD

    run_on_sample_selection = sample_filter is None or len(sample_filter) > THRESH

    run_on_otu_selection = otu_filter is None or len(otu_filter) > THRESH

    run_on_otu_comp = (
                    (otu_compliment is not None and len(otu_compliment) > THRESH) and
                    (sample_filter is None or len(sample_filter) > THRESH)
                    )

    run_on_sample_comp = (
                    (sample_compliment is not None and len(sample_compliment) > THRESH) and
                    (otu_filter is None or (otu_filter) > THRESH)
                    )

    num_total_samples = len(samples)
    num_total_otus = len(otus)

    if run_on_sample_selection and run_on_otu_selection:
        step = __ctwc_recursive__get_next_step(iteration_ind, 0)
        title = iteration_to_title(step)
        result, step_samp_filter, step_samp_compliment, p_vals, dist = run_iteration(title, "Pick samples...",
                                                                       data,
                                                                       tree,
                                                                       samples,
                                                                       otus,
                                                                       otu_filter,
                                                                       sample_filter,
                                                                       table,
                                                                       False,
                                                                       0 if sample_filter is None else len(sample_filter))
        sel_dist, ref_dist = dist
        iteration_results[title] = (result, p_vals, sel_dist, ref_dist, len(step_samp_filter), num_total_samples)
        if sample_filter is None or (len(step_samp_filter) > 0 and len(step_samp_filter) < len(sample_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       otu_filter,
                                       otu_compliment,
                                       step_samp_filter,
                                       step_samp_compliment,
                                       step,
                                       iteration_results)

    if run_on_otu_selection and run_on_sample_selection:
        step = __ctwc_recursive__get_next_step(iteration_ind, 1)
        title = iteration_to_title(step)
        result, step_otu_filter, step_otu_compliment, p_vals, dist = run_iteration(title, "Pick OTUs...",
                                                                     data,
                                                                     tree,
                                                                     samples,
                                                                     otus,
                                                                     otu_filter,
                                                                     sample_filter,
                                                                     table,
                                                                     True,
                                                                     0 if otu_filter is None else len(otu_filter))
        sel_dist, ref_dist = dist
        iteration_results[title] = (result, p_vals, sel_dist, ref_dist, len(step_otu_filter), num_total_otus)
        if otu_filter is None or (len(step_otu_filter) > 0 and len(step_otu_filter) < len(otu_filter)):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       step_otu_filter,
                                       step_otu_compliment,
                                       sample_filter,
                                       sample_compliment,
                                       step,
                                       iteration_results)

    if run_on_sample_comp and run_on_otu_selection:
        step = __ctwc_recursive__get_next_step(iteration_ind, 2)
        title = iteration_to_title(step)
        result, step_samp_filter, step_samp_compliment, p_vals, dist = run_iteration(title, "Pick samples from compliment...",
                                                                        data,
                                                                        tree,
                                                                        samples,
                                                                        otus,
                                                                        otu_filter,
                                                                        sample_compliment,
                                                                        table,
                                                                        False)
        sel_dist, ref_dist = dist
        iteration_results[title] = (result, p_vals, sel_dist, ref_dist, len(step_samp_filter), num_total_samples)
        if len(step_samp_filter) < len(sample_compliment) and len(step_samp_filter) > 0:
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       otu_filter,
                                       otu_compliment,
                                       step_samp_filter,
                                       step_samp_compliment,
                                       step,
                                       iteration_results)

    if run_on_sample_comp:
        step = __ctwc_recursive__get_next_step(iteration_ind, 5)
        title = iteration_to_title(step)
        result, step_otu_filter, step_otu_compliment, p_vals, dist = run_iteration(title, "Pick OTUs from samples compliment...",
                                                                        data,
                                                                        tree,
                                                                        samples,
                                                                        otus,
                                                                        None,
                                                                        sample_compliment,
                                                                        table,
                                                                        True)
        sel_dist, ref_dist = dist
        iteration_results[title] = (result, p_vals, sel_dist, ref_dist, len(step_otu_filter), num_total_otus)
        if otu_filter is None or (len(step_otu_filter) < len(otu_filter) and len(step_otu_filter) > 0):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       step_otu_filter,
                                       step_otu_compliment,
                                       sample_filter,
                                       sample_compliment,
                                       step,
                                       iteration_results)


    if run_on_otu_comp and run_on_sample_selection:
        step = __ctwc_recursive__get_next_step(iteration_ind, 3)
        title = iteration_to_title(step)
        result, step_otu_filter, step_otu_compliment, p_vals, dist = run_iteration(title, "Pick OTUs from compliment...",
                                                                        data,
                                                                        tree,
                                                                        samples,
                                                                        otus,
                                                                        otu_compliment,
                                                                        sample_filter,
                                                                        table,
                                                                        True)
        sel_dist, ref_dist = dist
        iteration_results[title] = (result, p_vals, sel_dist, ref_dist, len(step_otu_filter), num_total_otus)
        if len(step_otu_filter) < len(otu_compliment) and len(step_otu_filter) > 0:
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       step_otu_filter,
                                       step_otu_compliment,
                                       sample_filter,
                                       sample_compliment,
                                       step,
                                       iteration_results)

    if run_on_otu_comp:
        step = __ctwc_recursive__get_next_step(iteration_ind, 4)
        title = iteration_to_title(step)
        result, step_samp_filter, step_samp_compliment, p_vals, dist = run_iteration(title, "Pick samples from OTUs compliment...",
                                                                        data,
                                                                        tree,
                                                                        samples,
                                                                        otus,
                                                                        otu_compliment,
                                                                        None,
                                                                        table,
                                                                        False)
        sel_dist, ref_dist = dist
        iteration_results[title] = (result, p_vals, sel_dist, ref_dist, len(step_samp_filter), num_total_samples)
        if sample_filter is None or (len(step_samp_filter) < len(sample_filter) and len(step_samp_filter) > 0):
            __ctwc_recursive_iteration(data, tree, samples, otus, table,
                                       otu_filter,
                                       otu_compliment,
                                       step_samp_filter,
                                       step_samp_compliment,
                                       step,
                                       iteration_results)


def test():
    try:
        ctwc__plot.init()
        np.seterr(all="ignore")
        samples, otus, tree, data, table = ctwc__distance_matrix.get_data(use_real_data=True, full_set=False)

        output = ctwc_recursive_select(data, tree, samples, otus, table)
        INFO("Full data size: {0} X {1}".format(data.shape[0], data.shape[1]))
        for elem in output:
            keys, pv = get_top_p_val(output[elem][RES_IND_P_VAL])
            INFO("{0}: {1} X {2} - P Value {3} Keys {4}".format(elem, output[elem][RES_IND_INPUT][0], output[elem][RES_IND_INPUT][1], pv, keys))
        write_cluster_summary_for_all_files_in_path(CLUSTER_OUTPUT_FILE)
    except Exception as ex:
        ERROR("Failed with exception: {}, stack trace".format(str(ex)))
        ERROR("Calling stack:")
        ERROR(traceback.print_stack())

if __name__ == "__main__":
    test()
