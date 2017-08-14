#!/usr/bin/python
from ctwc__common import *
import ctwc__plot
import warnings
import numpy as np

RAW_MILK_FILES = [ 'milk_3572_otu_table.json', 'milk_3573_otu_table.json', 'milk_3574_otu_table.json', 'milk_3575_otu_table.json',
'milk_3576_otu_table.json' ] #, 'milk_3579_otu_table.json' ]
FILTERED_NORMALIZED_RAW_MILK_FILES = [ 'subset-normalized-milk_3572_otu_table.json', 'subset-normalized-milk_3574_otu_table.json', 'subset-normalized-milk_3576_otu_table.json',
                            'subset-normalized-milk_3573_otu_table.json', 'subset-normalized-milk_3575_otu_table.json', 'subset-normalized-milk_3579_otu_table.json' ]
FILTERED_RAW_MILK_FILES = [ 'subset-milk_3572_otu_table.json', 'subset-milk_3574_otu_table.json', 'subset-milk_3576_otu_table.json',
                            'subset-milk_3573_otu_table.json', 'subset-milk_3575_otu_table.json', 'subset-milk_3579_otu_table.json' ]
NORMALIZED_RAW_MILK_FILES = [ 'normalized-milk_3572_otu_table.json', 'normalized-milk_3574_otu_table.json', 'normalized-milk_3576_otu_table.json',
                            'normalized-milk_3573_otu_table.json',
                            'normalized-milk_3575_otu_table.json']# 'normalized-milk_3579_otu_table.json' ]
DENOVO_REPROCESSED_MILK_FILES = [ 'milk-sub15k-min10.json' ]

DENOVO_REPROCESSED_TWINS_FILES = [ 'twins-sub15k-min10.json' ]

#BIOM_FILES_DICT = DENOVO_REPROCESSED_MILK_FILES
BIOM_FILES_DICT = DENOVO_REPROCESSED_TWINS_FILES
#BIOM_FILES_DICT = FILTERED_RAW_MILK_FILES
#BIOM_FILES_DICT = FILTERED_NORMALIZED_RAW_MILK_FILES
#BIOM_FILES_DICT = RAW_MILK_FILES
#BIOM_FILES_DICT = NORMALIZED_RAW_MILK_FILES

RAW_MILK_TREE_FILE = "97_otus.tree"
DENOVO_REPROCESSED_MILK_TREE_FILE = "milk_all.tre"
DENOVO_REPROCESSED_TWINS_TREE_FILE = "twins_all.tre"

#TREE_FILE = DENOVO_REPROCESSED_MILK_TREE_FILE
TREE_FILE = DENOVO_REPROCESSED_TWINS_TREE_FILE
#TREE_FILE = RAW_MILK_TREE_FILE

def __get_default_tree(otus):
    from cogent.parse.tree import DndParser
    from cogent.maths.unifrac.fast_tree import UniFracTreeNode
    tree_str = "(((A:0.15)B:0.2,(C:0.3,D:0.4)E:0.6)F:0.1)G;"
    for i, otu in enumerate(otus):
        tree_str = tree_str.replace(chr(ord('A') + i), otu)
    tr = DndParser(tree_str, UniFracTreeNode)
    return tr

def __get_default_samples():
    return np.array(['mouth', 'leg', 'lips', 'armpit', 'foot'])

def __get_default_otus():
    return np.array(['donatello', 'leonardo', 'raphael', 'michelangelo', 'splinter', 'shredder', 'april'])

def __get_default_data(otus, samples):
    #return __get_randomly_pre_generated_data()
    return __get_synthetic_cluster_data(otus, samples)

def __get_synthetic_cluster_data(otus, samples):
    data = __get_randomly_pre_generated_data() # noise
    data += np.array([   [5, 13, 0, 0, 0],
                         [8, 15, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    return data

def __get_randomly_generated_data(otus, samples):
    num_otu = len(otus)
    num_samp = len(samples)
    data = np.random.randint(low=0, high=2, size=(num_otu, num_samp))
    return data

def __get_randomly_pre_generated_data():
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
    tr = get_tree_from_file(TREE_FILE)
    return tr

def get_biom_table_from_file(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from biom import parse_table
    with open(path) as f:
        table = parse_table(f)
        return table
    return None

def __add_suffix_to_sample_ids(table, suffix):
    for ind, samp in enumerate(table._sample_ids):
        samp_ = samp + suffix
        table._sample_ids[ind] = samp_
        table._sample_index[samp_] = table._sample_index[samp]
        table._sample_index.pop(samp, None)

def get_sample_biom_table(full_set=True):
    if not full_set:
        tables = [ get_biom_table_from_file(BIOM_FILES_DICT[0]) ]
    else:
        tables = map(get_biom_table_from_file, BIOM_FILES_DICT)
    table = tables[0]
    for ind, tab in enumerate(tables, 1):
        INFO("Dataset part {0} size: {1}".format(ind, tab.shape))
        if ind != 1:
            table = table.merge(tab)
    INFO("Complete dataset size: {0}".format(table.shape))
    return table.matrix_data.todense(), table.ids('observation'), table.ids('sample'), table

def __shuffle_2d_dist_matrix(mat):
    ASSERT(mat.shape[0] == mat.shape[1])
    rand_state = np.random.get_state()
    np.random.shuffle(mat)
    mat = mat.transpose()
    np.random.set_state(rand_state)
    np.random.shuffle(mat)
    return mat


def get_synthetic_biom_table(full_set=True):
    INFO("Synthesizing data based on sample table...")
    INFO("Reading sample files...")
    data, otus, samples, table = get_sample_biom_table(full_set)
    INFO("Generating new patterns in data...")
    # noise
    data = abs(np.random.normal(0, 5, size=data.shape)) / 100.0
    data[data < 0] = 0.0
    # cluster of samples
    size_of_samp_set = 150 if not full_set else 800
    size_of_otu_set = 5000 if not full_set else 10000
    size_of_partial_samp_set = 70 if not full_set else 300
    data[:, :size_of_samp_set] = 0.0
    data[100:size_of_otu_set, :size_of_samp_set] = 6.0 / 100.0 # 4900 OTU types common for 150 first samples
    # second cluster of samples
    data[size_of_otu_set + 100 : size_of_otu_set + 1000,
         size_of_partial_samp_set : size_of_samp_set] = 5.0 / 100.0 # 2900 OTU types common for the latter 80 of the first 150 samples
    ctwc__plot.plot_mat(data, header="Original Data Pre-shuffle")
    # shuffle along first axis:
    np.random.shuffle(data)
    # shuffle along second axis:
    data = data.transpose()
    np.random.shuffle(data)
    # transpose back:
    data = data.transpose()
    ctwc__plot.plot_mat(data, header="Original Data Post-shuffle")
    INFO("Done preparing synthetic data")
    return data, otus, samples, table

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

def test():
    samples = __get_default_samples()
    otus = __get_default_otus()
    tree = __get_default_tree(otus)
    data = __get_default_data(otus, samples)
    DEBUG("Tree:\n" + tree.asciiArt())
    DEBUG("Samples:\n" + str(samples))
    DEBUG("OTUs:\n" + str(otus))
    DEBUG("Data:\n" + str(data))
    get_sample_biom_table()

if __name__ == "__main__":
    test()
