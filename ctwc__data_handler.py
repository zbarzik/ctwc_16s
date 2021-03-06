#!/usr/bin/python
from ctwc__common import *
import ctwc__plot
import warnings
import numpy as np

DATASET = "milk" #"ag" # "cows" # "milk" # "twins"

RAW_MILK_FILES = [ 'milk_3572_otu_table.json', 'milk_3573_otu_table.json', 'milk_3574_otu_table.json', 'milk_3575_otu_table.json',
'milk_3576_otu_table.json' ] #, 'milk_3579_otu_table.json' ]
FILTERED_NORMALIZED_RAW_MILK_FILES = [ 'subset-normalized-milk_3572_otu_table.json', 'subset-normalized-milk_3574_otu_table.json', 'subset-normalized-milk_3576_otu_table.json',
                            'subset-normalized-milk_3573_otu_table.json', 'subset-normalized-milk_3575_otu_table.json', 'subset-normalized-milk_3579_otu_table.json' ]
FILTERED_RAW_MILK_FILES = [ 'subset-milk_3572_otu_table.json', 'subset-milk_3574_otu_table.json', 'subset-milk_3576_otu_table.json',
                            'subset-milk_3573_otu_table.json', 'subset-milk_3575_otu_table.json', 'subset-milk_3579_otu_table.json' ]
NORMALIZED_RAW_MILK_FILES = [ 'normalized-milk_3572_otu_table.json', 'normalized-milk_3574_otu_table.json', 'normalized-milk_3576_otu_table.json',
                            'normalized-milk_3573_otu_table.json',
                            'normalized-milk_3575_otu_table.json']# 'normalized-milk_3579_otu_table.json' ]

#AG_RAW_FILES = [ 'new_data/AG_even1k_gut_only.json' ]
AG_RAW_FILES = [ 'new_data/AG_even1k.json' ]

DENOVO_REPROCESSED_MILK_FILES = [ 'milk-sub15k-min10.json' ]

DENOVO_REPROCESSED_TWINS_FILES = [ 'twins-sub15k-min10.json' ]

RAW_COWS_BIOM_FILES = [ 'Core_58%_otu_table.json' ]

#BIOM_FILES_DICT = FILTERED_RAW_MILK_FILES
#BIOM_FILES_DICT = FILTERED_NORMALIZED_RAW_MILK_FILES
#BIOM_FILES_DICT = RAW_MILK_FILES
#BIOM_FILES_DICT = NORMALIZED_RAW_MILK_FILES

RAW_MILK_TREE_FILE = "97_otus.tree"
AG_RAW_TREE_FILE = "97_otus.tree"
DENOVO_REPROCESSED_MILK_TREE_FILE = "milk_all.tre"
DENOVO_REPROCESSED_TWINS_TREE_FILE = "twins_all.tre"
COWS_TREE_FILE = "unified_seq_run.fq_rep_set.fasta_mafft2_no_bc_otus_pfiltered.tre"


TREE_FILE = DENOVO_REPROCESSED_MILK_TREE_FILE
#TREE_FILE = DENOVO_REPROCESSED_TWINS_TREE_FILE
#TREE_FILE = RAW_MILK_TREE_FILE

if DATASET == "milk":
    BIOM_FILES_DICT = DENOVO_REPROCESSED_MILK_FILES
    TREE_FILE = DENOVO_REPROCESSED_MILK_TREE_FILE
elif DATASET == "twins":
    BIOM_FILES_DICT = DENOVO_REPROCESSED_TWINS_FILES
    TREE_FILE = DENOVO_REPROCESSED_TWINS_TREE_FILE
elif DATASET == "cows":
    BIOM_FILES_DICT = RAW_COWS_BIOM_FILES
    TREE_FILE = COWS_TREE_FILE
elif DATASET == "ag":
    BIOM_FILES_DICT = AG_RAW_FILES
    TREE_FILE = AG_RAW_TREE_FILE
else:
    FATAL("Unknown dataset")

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
    INFO("Using {0} dataset".format(DATASET))
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

def get_synthetic_small_input_biom_table_jagged():
    INFO("Syntherizing a small data set based on sample table...")
    INFO("Reading sample files...")
    num_otus = 400
    num_samples = 350
    data, otus, samples, table = get_sample_biom_table(False)
    INFO("Reducing data set size to ({},{})".format(num_otus, num_samples))
    otus = otus[:num_otus]
    samples = samples[:num_samples]
    table._data = table._data[:num_otus, :num_samples]
    data = abs(np.random.normal(0, 2, size=(num_otus,num_samples)))
    data[data < 0] = 0
    data[:] = 0.0
    # cluster of samples
    samp_set_1 = 100
    samp_set_2 = 110
    samp_set_3 = 120
    samp_set_4 = 130
    otu_set_1 = 140
    otu_set_2 = 150
    otu_set_3 = 160
    otu_set_4 = 170
    #data[:, :samp_set_1+samp_set_4] = 0.0
    #data[:otu_set_1+otu_set_4, :] = 0.0
    data[:otu_set_4, :samp_set_4] = 13.0
    data[otu_set_4:otu_set_4+otu_set_1, :samp_set_2] = 24.0
    data[:otu_set_2, samp_set_4:samp_set_4+samp_set_1] = 36.0
    data[otu_set_4:otu_set_4+otu_set_3, samp_set_4:samp_set_4+samp_set_3] = 44.0
    data = data / 100.0
    ctwc__plot.plot_mat(data, header="Original Data Pre-shuffle")
    # shuffle along first axis:
    #np.random.shuffle(data)
    # shuffle along second axis:
    #data = data.transpose()
    #np.random.shuffle(data)
    # transpose back:
    #data = data.transpose()
    #ctwc__plot.plot_mat(data, header="Original Data Post-shuffle")
    INFO("Done preparing synthetic data")
    return data, otus, samples, table


def get_synthetic_biom_table_jagged(full_set=True):
    if not full_set:
        return get_synthetic_small_input_biom_table_jagged()
    INFO("Synthesizing data based on sample table...")
    INFO("Reading sample files...")
    data, otus, samples, table = get_sample_biom_table(full_set)
    INFO("Generating jagged checkerbox patterns in data...")
    # noise
    data = abs(np.random.normal(0, 2, size=data.shape)) / 100.0
    data[data < 0] = 0
    # cluster of samples
    samp_set_1 = 500
    samp_set_2 = 550
    samp_set_3 = 400
    samp_set_4 = 600
    otu_set_1 = 5000
    otu_set_2 = 4000
    otu_set_3 = 7500
    otu_set_4 = 4500
    data[:, :samp_set_1+samp_set_2] = 0.0
    data[:otu_set_1, :samp_set_1] = 32.0 / 100.0
    data[otu_set_1:otu_set_1+otu_set_2, :samp_set_3] = 24.0 / 100.0
    data[:otu_set_3, samp_set_1:samp_set_1+samp_set_2] = 36.0 / 100.0
    data[otu_set_3:otu_set_3+otu_set_4, samp_set_3:samp_set_3+samp_set_4] = 44.0 / 100.0
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

def get_synthetic_biom_table_single_axis_noise(full_set=True):
    INFO("Synthesizing data based on sample table...")
    INFO("Reading sample files...")
    data, otus, samples, table = get_sample_biom_table(full_set)
    INFO("Generating single axis noise patterns in data...")
    # noise
    data = abs(np.random.normal(0, 5, size=data.shape)) / 100.0
    data[data < 0] = 0.0
    # cluster of samples
    size_of_samp_set = 150 if not full_set else 800
    size_of_otu_set = 5000 if not full_set else 8000
    size_of_partial_otu_set = 2000
    size_of_partial_samp_set = 70 if not full_set else 300
    data[:, :size_of_samp_set] = 0.0
    data[100:size_of_otu_set, :size_of_samp_set] = 6.0 / 100.0 # 7900 OTU types common for 150 first samples
    # second cluster of samples
    data[size_of_otu_set + 100 : size_of_otu_set + size_of_partial_otu_set,
         size_of_partial_samp_set : size_of_samp_set] = 5.0 / 100.0 # 1900 OTU types common for the latter 500 of the first 800 samples
    ctwc__plot.plot_mat(data, header="Original Data Pre-shuffle")
    # shuffle along first axis:
    #np.random.shuffle(data)
    # shuffle along second axis:
    #data = data.transpose()
    #np.random.shuffle(data)
    # transpose back:
    #data = data.transpose()
    ctwc__plot.plot_mat(data, header="Original Data Post-shuffle")
    INFO("Done preparing synthetic data")
    return data, otus, samples, table


def get_samples_by_indices(indices, table):
    s_indices = set(indices)
    samples = [ str(samp) for samp in table._sample_index if table._sample_index[samp] in s_indices ]
    return samples

def get_otus_by_indices(indices, table):
    s_indices = set(indices)
    otus = [ str(obs) for obs in table._obs_index if table._obs_index[obs] in s_indices ]
    return otus

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
