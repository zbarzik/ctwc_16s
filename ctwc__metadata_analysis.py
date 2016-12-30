#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP,has_value
import csv, bisect, math, scipy, scipy.stats

TAXA_LINE_STRUCUTURE = ["otu", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
SAMPLES_LINE_STRUCTURE = ['sample_name', 'bactoscan', 'check_in_time', 'check_out_time', 'collection_timestamp', 'date_pcr',
'dateprocessed', 'description', 'dmissing_extraction', 'dna_extracted', 'dna_extraction', 'elevation', 'env_biome',
'env_feature', 'env_matter', 'env_package', 'geo_loc_name', 'host_subject_id', 'investigation_type', 'latitude', 'longitude',
'notes', 'physical_specimen_location', 'physical_specimen_remaining', 'pma_treatment', 'sample_type', 'scientific_name',
'season', 'silo_lot_id', 'tanker_cip_date', 'tanker_cip_time', 'taxon_id', 'title']

SAMPLES_MD_FILE = "10485_20160420-093403.txt"
TAXA_MD_FILE = "97_otu_taxonomy.txt"
TEST = False

def __generate_histogram_for_taxonomy_rank(rank, taxa):
    d = dict()
    ind = TAXA_LINE_STRUCUTURE.index(rank)
    for e in taxa:
        entry = e.split()
        if not d.has_key(entry[ind].strip()):
            d[entry[ind]] = 1
        else:
            d[entry[ind]] += 1
    return d

def get_taxonomies_for_otus(otus):
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    taxa = []
    otus_s = sorted(otus)
    for line in lines:
        if has_value(otus_s, line.split()[0].strip()):
            taxa.append(line)
    return taxa

def calculate_otus_histogram(otus_list):
    taxa = get_taxonomies_for_otus(otus_list)
    hist = dict()
    for rank in TAXA_LINE_STRUCUTURE[1:]:
        hist[rank] = __generate_histogram_for_taxonomy_rank(rank, taxa)
    return hist

def get_collection_dates_for_samples(samples):
    return get_field_for_samples("collection_timestamp", samples)

def get_field_for_samples(field, samples):
    out = []
    samples_s = sorted(samples)
    field_index = SAMPLES_LINE_STRUCTURE.index("collection_timestamp")
    sample_index = SAMPLES_LINE_STRUCTURE.index("sample_name")
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if has_value(samples_s, row[sample_index].strip()):
                field = row[field_index].strip()
                out.append(field)
    return out

def get_metadata_for_samples(samples_list):
    out = []
    samples_s = sorted(samples_list)
    sample_index = SAMPLES_LINE_STRUCTURE.index("sample_name")
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if has_value(samples_s, row[sample_index].strip()):
                out.append(row)
    return out

def __generate_histogram_for_samples_field(field, samples_md):
    d = dict()
    ind = SAMPLES_LINE_STRUCTURE.index(field)
    for entry in samples_md:
        if not d.has_key(entry[ind]):
            d[entry[ind]] = 1
        else:
            d[entry[ind]] += 1
    return d

def calculate_samples_histogram(samples_list):
    md = get_metadata_for_samples(samples_list)
    hist = dict()
    for field in SAMPLES_LINE_STRUCTURE[1:]:
        hist[field] = __generate_histogram_for_samples_field(field, md)
    return hist

def calculate_samples_distribution(samples_list):
    samp_hist = calculate_samples_histogram(samples_list)
    return __calculate_samples_distribution_from_histogram(samp_hist)

def __calculate_filtered_samples_distribution(samples_list, field, filt):
    samp_hist = calculate_samples_histogram(samples_list)
    for k in samp_hist[field].keys():
        if k not in filt:
            samp_hist[field].pop(k)
    return __calculate_samples_distribution_from_histogram(samp_hist)

def __calculate_samples_distribution_from_histogram(samp_hist):
    percentiles = dict()
    for field in SAMPLES_LINE_STRUCTURE[1:]:
        total = 0.0
        for key in samp_hist[field].keys():
            total += samp_hist[field][key]
        ASSERT(total > 0) # can't happen - at least one key has to exist when iterating on samples
        percentiles[field] = ( total, { key: (100.0 * samp_hist[field][key])/total for key in samp_hist[field].keys() } )
        if TEST:
            s = 0
            for key in percentiles[field][1].keys():
                s += percentiles[field][1][key]
            ASSERT(round(s) == 100.0)
    return percentiles

def calculate_otus_distribution(otus_list):
    otus_hist = calculate_otus_histogram(otus_list)
    return __calculate_otus_distribution_from_histogram(otus_hist)

def __calculate_filtered_otus_distribution(otus_list, rank, filt):
    otus_hist = calculate_otus_histogram(otus_list)
    for k in otus_hist[rank].keys():
        if k not in filt:
            otus_hist[rank].pop(k)
    return __calculate_otus_distribution_from_histogram(otus_hist)

def __calculate_otus_distribution_from_histogram(otu_hist):
    percentiles = dict()
    for rank in TAXA_LINE_STRUCUTURE[1:]:
        total = 0.0
        for key in otu_hist[rank].keys():
            total += otu_hist[rank][key]
        ASSERT(total > 0) # can't happen - every OTU has to be a part of at least one classification
        percentiles[rank] = ( total, { key: (100.0 * otu_hist[rank][key])/total for key in otu_hist[rank].keys() } )
        if TEST:
            s = 0
            for key in percentiles[rank][1].keys():
                s += percentiles[rank][1][key]
            ASSERT(round(s) == 100.0)
    return percentiles

def __calculate_z_score(selected_num, selected_dist, general_dist):
    s_ = selected_dist / 100.0
    g_ = general_dist / 100.0
    n_ = selected_num
    try:
        z_ = (s_ - g_) / math.sqrt(g_ * (1 - g_) / n_)
    except:
        z_ = 1.0
    return z_

def __calculate_p_value(selected_num, selected_dist, general_dist):
    z = __calculate_z_score(selected_num, selected_dist, general_dist)
    p = scipy.stats.norm.sf(abs(z))
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
    total = sel_dist[key][0]
    pvals = dict()
    for k in ref_dist_k.keys():
        tmp[k] = g(sel_dist_k, k)
    for k in tmp.keys():
        pvals[k] = __calculate_p_value(total, tmp[k], ref_dist_k[k])
    return pvals

def calculate_otus_p_values(selection_distribution, reference_distribution):
    sel_dist = selection_distribution
    ref_dist = reference_distribution
    p_vals = dict()
    for rank in sel_dist.keys():
        p_vals[rank] = __calculate_otus_p_values_for_rank(sel_dist, ref_dist, rank)
    return p_vals

def calculate_samples_p_values(selection_distribution, reference_distribution):
    sel_dist = selection_distribution
    ref_dist = reference_distribution
    p_vals = dict()
    for field in sel_dist.keys():
        p_vals[field] = __calculate_samples_p_values_for_field(sel_dist, ref_dist, field)
    return p_vals

def test():
    globals()['TEST'] = True
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    otus_list = map(lambda x: x.split()[0].strip(), lines)
    otus_dist = calculate_otus_distribution(otus_list)
    filt = ['s__princeps', 's__nodulosa']
    otus_dist_1 = __calculate_filtered_otus_distribution(otus_list, 'species', filt)
    p_vals = calculate_otus_p_values(otus_dist_1, otus_dist)

    with open(SAMPLES_MD_FILE, 'r') as samp_fn:
        lines = samp_fn.readlines()
    samples_list = map(lambda x: x.split()[0].strip(), lines)[1:]
    samples_dist = calculate_samples_distribution(samples_list)
    samples_dist_1 = __calculate_filtered_samples_distribution(samples_list, 'season', ['spring', 'fall', 'summer'])
    p_vals = calculate_samples_p_values(samples_dist_1, samples_dist)
    BP()

if __name__ == "__main__":
    test()
