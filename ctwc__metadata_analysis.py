#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP,has_value
import csv, bisect, math, scipy, scipy.stats, random

TAXA_LINE_STRUCUTURE = ["otu", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
SAMPLES_LINE_STRUCTURE = ['sample_name', 'bactoscan', 'check_in_time', 'check_out_time', 'collection_timestamp', 'date_pcr',
'dateprocessed', 'description', 'dmissing_extraction', 'dna_extracted', 'dna_extraction', 'elevation', 'env_biome',
'env_feature', 'env_matter', 'env_package', 'geo_loc_name', 'host_subject_id', 'investigation_type', 'latitude', 'longitude',
'notes', 'physical_specimen_location', 'physical_specimen_remaining', 'pma_treatment', 'sample_type', 'scientific_name',
'season', 'silo_lot_id', 'tanker_cip_date', 'tanker_cip_time', 'taxon_id', 'title']
SAMPLES_SKIP_FIELDS = ['check_in_time', 'collection_timestamp', 'date_pcr', 'latitude', 'longitude', 'notes', 'physical_specimen_remaining', 'title']
SAMPLES_MD_FILE = "10485_20160420-093403.txt"
TAXA_MD_FILE = "97_otu_taxonomy.txt"
TEST = False

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

def calculate_otus_histogram(otus_list, filt=None, filt_rank='species'):
    taxa = get_taxonomies_for_otus(otus_list)
    hist = dict()
    for rank in TAXA_LINE_STRUCUTURE[1:]:
        hist[rank] = __generate_histogram_for_taxonomy_rank(rank, taxa, filt, filt_rank)
    return hist

def get_collection_dates_for_samples(samples):
    return get_field_for_samples("collection_timestamp", samples)

def get_field_for_samples(field, samples):
    out = []
    samples_s = set(samples)
    field_index = SAMPLES_LINE_STRUCTURE.index("collection_timestamp")
    sample_index = SAMPLES_LINE_STRUCTURE.index("sample_name")
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if row[sample_index].strip() in samples_s:
                field = row[field_index].strip()
                out.append(field)
    return out

def get_metadata_for_samples(samples_list):
    out = []
    samples_s = set(samples_list)
    sample_index = SAMPLES_LINE_STRUCTURE.index("sample_name")
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if row[sample_index].strip() in samples_s:
                out.append(row)
    return out

def __generate_histogram_for_samples_field(field, samples_md, filt=None, filt_field='bactoscan'):
    d = dict()
    ind = SAMPLES_LINE_STRUCTURE.index(field)
    filter_ind = SAMPLES_LINE_STRUCTURE.index(filt_field)
    for entry in samples_md:
        if filt is not None:
            if entry[filter_ind] not in filt:
                continue
        d[entry[ind]] = d[entry[ind]] + 1 if d.has_key(entry[ind]) else 1
    return d

def calculate_samples_histogram(samples_list, filt=None, filt_field='bactoscan'):
    md = get_metadata_for_samples(samples_list)
    hist = dict()
    for field in SAMPLES_LINE_STRUCTURE[1:]:
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
    for field in SAMPLES_LINE_STRUCTURE[1:]:
        total = 0.0
        if field not in samp_hist.keys():
            continue
        for key in samp_hist[field].keys():
            total += samp_hist[field][key]
        ASSERT(total > 0) # can't happen - at least one key has to exist when iterating on samples
        percentiles[field] = ( total, { key: samp_hist[field][key]/total for key in samp_hist[field].keys() } )
        if TEST:
            s = 0
            for key in percentiles[field][1].keys():
                s += percentiles[field][1][key]
            ASSERT(round(s) == 1.0)
    return percentiles

def calculate_otus_distribution(otus_list):
    otus_hist = calculate_otus_histogram(otus_list)
    return __calculate_otus_distribution_from_histogram(otus_hist)

def __calculate_filtered_otus_distribution(otus_list, filt, filt_rank):
    otus_hist = calculate_otus_histogram(otus_list, set(filt), filt_rank)
    return __calculate_otus_distribution_from_histogram(otus_hist)

def __calculate_otus_distribution_from_histogram(otu_hist):
    percentiles = dict()
    for rank in TAXA_LINE_STRUCUTURE[1:]:
        total = 0.0
        for key in otu_hist[rank].keys():
            total += otu_hist[rank][key]
        ASSERT(total > 0) # can't happen - every OTU has to be a part of at least one classification
        percentiles[rank] = ( total, { key: otu_hist[rank][key]/total for key in otu_hist[rank].keys() } )
        if TEST:
            s = 0
            for key in percentiles[rank][1].keys():
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
    for k in ref_dist_k.keys():
        tmp[k] = g(sel_dist_k, k)
    for k in tmp.keys():
        pvals[k] = __calculate_p_value(total, selected_size, tmp[k], ref_dist_k[k])
    return pvals

def calculate_otus_p_values(selection_distribution, reference_distribution):
    sel_dist = selection_distribution
    ref_dist = reference_distribution
    p_vals = dict()
    for rank in sel_dist.keys():
        if len(ref_dist[rank][1].keys()) == 1:
            continue # skip if there's no dynamic range
        p_vals[rank] = __calculate_otus_p_values_for_rank(sel_dist, ref_dist, rank)
    return p_vals

def calculate_samples_p_values(selection_distribution, reference_distribution):
    sel_dist = selection_distribution
    ref_dist = reference_distribution
    p_vals = dict()
    for field in sel_dist.keys():
        if len(ref_dist[field][1].keys()) == 1:
            continue # skip if there's no dynamic range
        p_vals[field] = __calculate_samples_p_values_for_field(sel_dist, ref_dist, field)
    return p_vals

def __prepare_p_val_vec(p_vals):
    vec = []
    for k1 in p_vals.keys():
        for k2 in p_vals[k1].keys():
            vec.append(p_vals[k1][k2])
    return vec

def __correct_p_vals(q_vals_vec, p_vals):
    ind = 0
    for k1 in p_vals.keys():
        for k2 in p_vals[k1].keys():
            p_vals[k1][k2] = q_vals_vec[ind]
            ind += 1
    return p_vals

def __corrected_p_values(p_vals_vec):
    import statsmodels.sandbox.stats.multicomp
    return statsmodels.sandbox.stats.multicomp.multipletests(p_vals_vec)

def correct_p_vals(p_vals):
    p_vals_vec = __prepare_p_val_vec(p_vals)
    q_vals_vec = __corrected_p_values(p_vals_vec)
    return __correct_p_vals(q_vals_vec[1], p_vals)

def test():
    globals()['TEST'] = True
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    otus_list = [ x.split()[0].strip() for x in lines ]
    otus_dist = calculate_otus_distribution(otus_list)
    filt = ['o__Clostridiales']
    otus_dist_1 = __calculate_filtered_otus_distribution(otus_list, filt, 'order')
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


if __name__ == "__main__":
    test()
