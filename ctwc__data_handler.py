#!/usr/bin/python
from ctwc__common import ASSERT,DEBUG,INFO,WARN,ERROR,FATAL,BP
import warnings

import numpy as np

def get_default_tree(otus):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    tree_str = "(((A:0.15)B:0.2,(C:0.3,D:0.4)E:0.6)F:0.1)G;"
    for i, otu in enumerate(otus):
        tree_str = tree_str.replace(chr(ord('A') + i), otu)
    tr = DndParser(tree_str, UniFracTreeNode)
    return tr

def get_default_samples():
    return np.array(['mouth', 'leg', 'lips', 'armpit', 'foot'])

def get_default_otus():
    return np.array(['donatello', 'leonardo', 'raphael', 'michelangelo', 'splinter', 'shredder', 'april'])

def get_default_data(otus, samples):
    #return get_randomly_pre_generated_data()
    return get_synthetic_cluster_data(otus, samples)

def get_synthetic_cluster_data(otus, samples):
    data = get_randomly_pre_generated_data() # noise
    data += np.array([   [5, 13, 0, 0, 0],
                         [8, 15, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    return data

def get_randomly_generated_data(otus, samples):
    num_otu = len(otus)
    num_samp = len(samples)
    data = np.random.randint(low=0, high=2, size=(num_otu, num_samp))
    return data

def get_randomly_pre_generated_data():
    # For deterministic testing - this was generated once using the above function
    data = np.array([   [0, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [0, 0, 1, 1, 0],
                        [1, 1, 0, 1, 1],
                        [1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 0],
                        [0, 0, 0, 1, 1]])
    return data

def get_tree_from_file(path):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    f = open(path, 'r')
    tr = DndParser(f.read(), UniFracTreeNode)
    return tr

def get_gg_97_otu_tree():
    tr = get_tree_from_file('97_otus.tree')
    return tr

def get_biom_table_from_file(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from biom import parse_table
    with open(path) as f:
        table = parse_table(f)
        return table
    return None

def add_suffix_to_sample_ids(table, suffix):
    for ind, samp in enumerate(table._sample_ids):
        samp_ = samp + suffix
        table._sample_ids[ind] = samp_
        table._sample_index[samp_] = table._sample_index[samp]
        table._sample_index.pop(samp, None)

def get_sample_biom_table():
    tables = []
    tables.append(get_biom_table_from_file('milk_3572_otu_table.json'))
    #tables.append(get_biom_table_from_file('milk_3573_otu_table.json'))
    #tables.append(get_biom_table_from_file('milk_3574_otu_table.json'))
    #tables.append(get_biom_table_from_file('milk_3575_otu_table.json'))
    #tables.append(get_biom_table_from_file('milk_3576_otu_table.json'))
    table = tables[0]
    for ind, tab in enumerate(tables, 1):
        INFO("Dataset part {0} size: {1}".format(ind, tab.shape))
        if ind != 1:
            table = table.merge(tab)
    INFO("Complete dataset size: {0}".format(table.shape))
    return table.matrix_data.todense(), table.ids('observation'), table.ids('sample'), table

def get_samples_by_indices(indices, table):
    samples = []
    def __get_sample_by_index(index, table):
        for samp in table._sample_index:
            if table._sample_index[samp] == index:
                return samp
        return None

    for index in indices:
        samples.append(__get_sample_by_index(index, table))
    return map(str, samples)

def get_otus_by_indices(indices, table):
    otus = []
    def __get_otu_by_index(index, table):
        for obs in table._obs_index:
            if table._obs_index[obs] == index:
                return obs
        return None

    for index in indices:
        otus.append(__get_otu_by_index(index, table))
    return map(str, otus)

def get_collection_dates_for_samples(samples):
    dates = []
    for samp in samples:
        dates.append(get_sample_colletion_date(samp))
    return dates

def get_sample_colletion_date(sample):
    import csv
    with open('10485_20160420-093403.txt', 'r') as md_file:
        metadata = csv.reader(md_file, delimiter='\t', quotechar='\'')
        timestamp_index = -1
        for row in metadata:
            if timestamp_index == -1:
                for index, item in enumerate(row):
                    if item == "collection_timestamp":
                        timestamp_index = index
                        break
            elif sample in row:
                ts = row[timestamp_index].strip()
                return ts
    return None

def get_taxonomies_for_otus(otus):
    taxonomies = []
    for otu in otus:
        taxonomies.append(get_otu_taxonomy(otu))
    return taxonomies

def get_otu_taxonomy(otu):
    with open('97_otu_taxonomy.txt', 'r') as tax_fn:
        lines = tax_fn.readlines()
    for line in lines:
        if int(line.split()[0]) == int(otu):
            return line


def test():
    samples = get_default_samples()
    otus = get_default_otus()
    tree = get_default_tree(otus)
    data = get_default_data(otus, samples)
    DEBUG("Tree:\n" + tree.asciiArt())
    DEBUG("Samples:\n" + str(samples))
    DEBUG("OTUs:\n" + str(otus))
    DEBUG("Data:\n" + str(data))
    get_sample_biom_table()


if __name__ == "__main__":
    test()
