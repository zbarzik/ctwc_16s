#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP
import csv, bisect

TAXA_LINE_STRUCUTURE = ["otu", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
SAMPLES_LINE_STRUCTURE = ['sample_name', 'bactoscan', 'check_in_time', 'check_out_time', 'collection_timestamp', 'date_pcr',
'dateprocessed', 'description', 'dmissing_extraction', 'dna_extracted', 'dna_extraction', 'elevation', 'env_biome',
'env_feature', 'env_matter', 'env_package', 'geo_loc_name', 'host_subject_id', 'investigation_type', 'latitude', 'longitude',
'notes', 'physical_specimen_location', 'physical_specimen_remaining', 'pma_treatment', 'sample_type', 'scientific_name',
'season', 'silo_lot_id', 'tanker_cip_date', 'tanker_cip_time', 'taxon_id', 'title']

SAMPLES_MD_FILE = "10485_20160420-093403.txt"
TAXA_MD_FILE = "97_otu_taxonomy.txt"

def __generate_histogram_for_taxonomy_rank(rank, taxa):
    d = dict()
    ind = TAXA_LINE_STRUCUTURE.index(rank)
    for e in taxa:
        entry = e.split()
        if not d.has_key(entry[ind]):
            d[entry[ind]] = 1
        else:
            d[entry[ind]] += 1
    return d

def __has_value(sorted_list, value):
    i = bisect.bisect_left(sorted_list, value)
    if i != len(sorted_list) and sorted_list[i] == value:
        return True
    return False

def get_taxonomies_for_otus(otus):
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    taxa = []
    otus_s = sorted(otus)
    for line in lines:
        if __has_value(otus_s, int(line.split()[0])):
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
    sample_index= SAMPLES_LINE_STRUCTURE.index("sample_name")
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if __has_value(samples_s, row[sample_index].strip()):
                field = row[field_index].strip()
                out.append(field)
    return out

def get_metadata_for_samples(samples_list):
    out = []
    samples_s = sorted(samples_list)
    sample_index= SAMPLES_LINE_STRUCTURE.index("sample_name")
    with open(SAMPLES_MD_FILE, 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        for row in metadata:
            if __has_value(samples_s, row[sample_index].strip()):
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

def calculate_sample_histogram(samples_list):
    md = get_metadata_for_samples(samples_list)
    hist = dict()
    for field in SAMPLES_LINE_STRUCTURE[1:]:
        hist[field] = __generate_histogram_for_samples_field(field, md)
    return hist

def test():
    with open(TAXA_MD_FILE, 'r') as tax_fn:
        lines = tax_fn.readlines()
    otus_list = map(lambda x: int(x.split()[0]), lines)
    otus_hist = calculate_otus_histogram(otus_list)

    with open(TAXA_MD_FILE, 'r') as samp_fn:
        lines = samp_fn.readlines()
    samples_list = map(lambda x: x.split()[0], lines)[1:]
    samples_hist = calculate_sample_histogram(samples_list)

if __name__ == "__main__":
    test()
