#!/usr/bin/python
from ctwc__common import *
from ctwc__data_handler import DATASET
import csv, bisect, math, scipy, scipy.stats, random

TAXA_LINE_STRUCUTURE = ['otu', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
MILK_SAMPLES_ID_FIELD = 'sample_name'
TWINS_SAMPLES_ID_FIELD = 'run_s'
SAMPLES_SKIP_FIELDS = ['host_subject_id']
MILK_SAMPLES_MD_FILE = '10485_20160420-093403.txt'
TWINS_SAMPLES_MD_FILE = 'SraRunTable.txt'
TAXA_MD_FILE = '97_otu_taxonomy.txt'
OTU_RANKS_TO_SKIP = ['kingdom']
TEST = False

if DATASET == "milk":
    SAMPLES_MD_FILE = MILK_SAMPLES_MD_FILE
    SAMPLES_ID_FIELD = MILK_SAMPLES_ID_FIELD
elif DATASET == "twins":
    SAMPLES_ID_FIELD = TWINS_SAMPLES_ID_FIELD
    SAMPLES_MD_FILE = TWINS_SAMPLES_MD_FILE
else:
    FATAL("Unknown dataset")

Q_VALUE_FILENAME = "q_vals_{0}_{1}.csv"

MIN_Q_VAL = 0.05

@memoize
def get_samples_line_structure():
    with open(SAMPLES_MD_FILE, 'r') as fn:
        lines = fn.readlines()
        return lines[0].lower().split()

"""
filt is used for testing, allowing only a defined collection of species.
Note that for performance, filt should be a set.
"""
def __generate_histogram_for_taxonomy_rank(rank, taxa, filt=None, filt_rank='species'):
    d = dict()
    ind = TAXA_LINE_STRUCUTURE.index(rank)
    filter_ind = TAXA_LINE_STRUCUTURE.index(filt_rank)
    for e in taxa:
        entry = [ en.strip(';') for en in e.split() ]
        if filt is not None:
            if entry[filter_ind] not in filt:
                continue
        d[entry[ind]] = d[entry[ind]] + 1 if d.has_key(entry[ind]) else 1
    return d

def get_taxonomies_for_otus(otus):
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    taxa = []
    otus_s = set(otus)
    for line in lines:
        if line.split()[0].strip() in otus_s:
            taxa.append(line)
    return taxa

def calculate_otus_histogram(otus_list, otus_indices, table, filt=None, filt_rank='species'):
    if table is None:
        taxa = get_taxonomies_for_otus(otus_list)
    else:
        taxa = get_taxa_by_otu_indices(otus_indices, table)
    hist = dict()
    for rank in TAXA_LINE_STRUCUTURE[1:]:
        if rank not in OTU_RANKS_TO_SKIP:
            hist[rank] = __generate_histogram_for_taxonomy_rank(rank, taxa, filt, filt_rank)
    return hist

def get_metadata_for_samples(samples_list):
    out = []
    samples_s = set(samples_list)
    sample_index = get_samples_line_structure().index(SAMPLES_ID_FIELD)
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if row[sample_index].strip() in samples_s:
                out.append(row)
    return out

def __generate_histogram_for_samples_field(field, samples_md, filt, filt_field):
    d = dict()
    ind = get_samples_line_structure().index(field)
    for entry in samples_md:
        if filt is not None: # used for testing only
            filter_ind = get_samples_line_structure().index(filt_field)
            if entry[filter_ind] not in filt:
                continue
        d[entry[ind]] = d[entry[ind]] + 1 if d.has_key(entry[ind]) else 1
    return d

def calculate_samples_histogram(samples_list, filt=None, filt_field='bactoscan'):
    md = get_metadata_for_samples(samples_list)
    hist = dict()
    for field in get_samples_line_structure()[1:]:
        if field not in SAMPLES_SKIP_FIELDS:
            hist[field] = __generate_histogram_for_samples_field(field, md, filt, filt_field)
    return hist

def calculate_samples_distribution(samples_list):
    samp_hist = calculate_samples_histogram(samples_list)
    return __calculate_samples_distribution_from_histogram(samp_hist)

def __calculate_filtered_samples_distribution(samples_list, filt, filt_field):
    samp_hist = calculate_samples_histogram(samples_list, set(filt), filt_field)
    return __calculate_samples_distribution_from_histogram(samp_hist)

def __calculate_samples_distribution_from_histogram(samp_hist):
    percentiles = dict()
    for field in get_samples_line_structure()[1:]:
        total = 0.0
        if field not in samp_hist:
            continue
        for key in samp_hist[field]:
            total += samp_hist[field][key]
        ASSERT(total > 0) # can't happen - at least one key has to exist when iterating on samples
        percentiles[field] = ( total, { key: samp_hist[field][key]/total for key in samp_hist[field].keys() } )
        if TEST:
            s = 0
            for key in percentiles[field][1]:
                s += percentiles[field][1][key]
            ASSERT(round(s) == 1.0)
    return percentiles

def calculate_otus_distribution(otus_list, otus_indices, table=None):
    otus_hist = calculate_otus_histogram(otus_list, otus_indices, table)
    return __calculate_otus_distribution_from_histogram(otus_hist)

def __calculate_filtered_otus_distribution(otus_list, otus_indices, filt, filt_rank):
    otus_hist = calculate_otus_histogram(otus_list, otus_indices, table, set(filt), filt_rank)
    return __calculate_otus_distribution_from_histogram(otus_hist)

def __calculate_otus_distribution_from_histogram(otu_hist):
    percentiles = dict()
    for rank in set(TAXA_LINE_STRUCUTURE[1:]) - set(OTU_RANKS_TO_SKIP):
        total = 0.0
        for key in otu_hist[rank]:
            total += otu_hist[rank][key]
        ASSERT(total > 0) # can't happen - every OTU has to be a part of at least one classification
        percentiles[rank] = ( total, { key: otu_hist[rank][key]/total for key in otu_hist[rank].keys() } )
        if TEST:
            s = 0
            for key in percentiles[rank][1]:
                s += percentiles[rank][1][key]
            ASSERT(round(s) == 1.0)
    return percentiles

def __calculate_p_value(total, selected_size, selected_dist, general_dist):
    M = total
    N = selected_size
    n = general_dist * total
    x = selected_dist * selected_size
    hg = scipy.stats.hypergeom(M=M, N=N, n=n)
    p = hg.sf(x)
    return p

def __calculate_otus_p_values_for_rank(sel_dist, ref_dist, rank):
    return __calculate_generic_p_values_for_key(sel_dist, ref_dist, rank)

def __calculate_samples_p_values_for_field(sel_dist, ref_dist, field):
    return __calculate_generic_p_values_for_key(sel_dist, ref_dist, field)

def __calculate_generic_p_values_for_key(sel_dist, ref_dist, key):
    tmp = dict()
    ref_dist_k = ref_dist[key][1]
    sel_dist_k = sel_dist[key][1]
    def g(d, k):
        return 0.0 if not d.has_key(k) else d[k]
    selected_size = sel_dist[key][0]
    total = ref_dist[key][0]
    pvals = dict()
    for k in ref_dist_k:
        tmp[k] = g(sel_dist_k, k)
    for k in tmp:
        pvals[k] = __calculate_p_value(total, selected_size, tmp[k], ref_dist_k[k])
    return pvals

def calculate_otus_p_values(selection_distribution, reference_distribution):
    sel_dist = selection_distribution
    ref_dist = reference_distribution
    p_vals = dict()
    for rank in sel_dist:
        if len(ref_dist[rank][1].keys()) == 1:
            continue # skip if there's no dynamic range
        p_vals[rank] = __calculate_otus_p_values_for_rank(sel_dist, ref_dist, rank)
    return p_vals

def calculate_samples_p_values(selection_distribution, reference_distribution):
    sel_dist = selection_distribution
    ref_dist = reference_distribution
    p_vals = dict()
    for field in sel_dist:
        if len(ref_dist[field][1].keys()) == 1:
            continue # skip if there's no dynamic range
        p_vals[field] = __calculate_samples_p_values_for_field(sel_dist, ref_dist, field)
    return p_vals

def __prepare_p_val_vec(p_vals):
    vec = []
    for k1 in p_vals:
        for k2 in p_vals[k1]:
            vec.append(p_vals[k1][k2])
    return vec

def __correct_p_vals(q_vals_vec, p_vals):
    ind = 0
    for k1 in p_vals:
        for k2 in p_vals[k1]:
            p_vals[k1][k2] = q_vals_vec[ind]
            ind += 1
    return p_vals

def __corrected_p_values(p_vals_vec):
    import statsmodels.sandbox.stats.multicomp
    return statsmodels.sandbox.stats.multicomp.multipletests(p_vals_vec)

def correct_p_vals(p_vals):
    p_vals_vec = __prepare_p_val_vec(p_vals)
    q_vals_vec = __corrected_p_values(p_vals_vec)
    q_vals = __correct_p_vals(q_vals_vec[1], p_vals)
    screened = {}
    for k1 in q_vals:
        for k2 in q_vals[k1]:
            if q_vals[k1][k2] < MIN_Q_VAL:
                if not screened.has_key(k1):
                    screened[k1] = {}
                screened[k1][k2] = q_vals[k1][k2]
    return screened

def save_q_values_to_csv_for_iteration(csv_writer, key, q_vals, sel_dist, ref_dist, num_selected, num_total):
    DEBUG("csv_writer {0}, key {1}, q_vals[key] {2}, sel_dist[key][1] {3}, ref_dist[key][1] {4}, num_selected {5}, num_total {6}".format(csv_writer, key, q_vals[key], sel_dist[key][1], ref_dist[key][1], num_selected, num_total))
    write_dict_entry_to_open_csv_file(csv_writer, key, q_vals[key], sel_dist[key][1], ref_dist[key][1], num_selected, num_total)

def get_taxa_by_otu_indices(indices, table):
    def fix_taxonomy_structure(taxonomy):
        fixed_taxonomy = {}
        for taxon in taxonomy:
            fixed_taxonomy[taxon[0]] = taxon
        for char in [ rank[0] for rank in TAXA_LINE_STRUCUTURE[1:] ]:
            if not fixed_taxonomy.has_key(char):
                fixed_taxonomy[char] = char + "__"
        fixed_taxonomy['o'] = 'o__'
        return [ fixed_taxonomy[ch] for ch in [ rank[0] for rank in TAXA_LINE_STRUCUTURE ] ]
    return [ "; ".join(fix_taxonomy_structure(table._observation_metadata[ind]['taxonomy'])) for ind in indices ]

def test():
    globals()['TEST'] = True
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    otus_list = [ x.split()[0].strip() for x in lines ]
    otus_dist = calculate_otus_distribution(otus_list)
    filt = ['o__Clostridiales']
    otus_dist_1 = __calculate_filtered_otus_distribution(otus_list, range(len(otus_list)), filt, 'order')
    p_vals = calculate_otus_p_values(otus_dist_1, otus_dist)
    q_vals = correct_p_vals(p_vals)

    with open(SAMPLES_MD_FILE, 'r') as samp_fn:
        lines = samp_fn.readlines()
    samples_list = [ x.split()[0].strip() for x in lines[1:]  ]
    samples_dist = calculate_samples_distribution(samples_list)
    samples_dist_1 = __calculate_filtered_samples_distribution(samples_list, ['spring', 'fall', 'summer'], 'season')
    p_vals = calculate_samples_p_values(samples_dist_1, samples_dist)

    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    otus_list = [ x.split()[0].strip() for x in lines if random.randint(0, 6) == 0 ]
    otus_dist_2 = calculate_otus_distribution(otus_list)
    p_vals = calculate_otus_p_values(otus_dist_2, otus_dist)
    q_vals = correct_p_vals(p_vals)

if __name__ == "__main__":
    test()
